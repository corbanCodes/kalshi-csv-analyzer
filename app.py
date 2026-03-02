"""
Kalshi CSV Analyzer Dashboard
Upload trade CSVs, filter bad trades, simulate martingale strategies, see projections.
"""

import os
import io
import uuid
import tempfile
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash
from functools import wraps
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')

# Password from environment variable
APP_PASSWORD = os.environ.get('ANALYZER_PASSWORD', 'kalshi2024')

# Upload storage - use temp directory
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'kalshi_analyzer_uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_upload_path(file_id):
    """Get path for uploaded file"""
    return os.path.join(UPLOAD_DIR, f'{file_id}.csv')

def load_trades_df():
    """Load trades dataframe from stored file"""
    file_id = session.get('file_id')
    if not file_id:
        return None
    filepath = get_upload_path(file_id)
    if not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath)

# =============================================================================
# AUTH
# =============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == APP_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            flash('Invalid password', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# =============================================================================
# TRADE PROCESSING
# =============================================================================

def filter_unprofitable_trades(df, max_entry_price=90):
    """Filter out trades above the entry price threshold (default 90c)"""
    return df[df['Entry Price'] <= max_entry_price].copy()

def simulate_strategy(trades_df, strategy='flat', base_bet=10, starting_bankroll=1000,
                      bankroll_pct=None, mart_start_after=1, max_multiplier=64):
    """
    Simulate a betting strategy on trades.

    strategy: 'flat', 'mart1', 'mart3', 'mart_custom', 'infinite_mart'
    base_bet: Base bet size in dollars
    starting_bankroll: Starting capital
    bankroll_pct: If set, bet this % of current bankroll instead of fixed
    mart_start_after: For mart_custom, start doubling after N consecutive losses
    max_multiplier: Cap multiplier (except for infinite_mart)
    """
    results = []
    bankroll = starting_bankroll
    current_mult = 1
    consec_losses = 0
    peak_bankroll = starting_bankroll
    max_drawdown = 0
    total_wagered = 0

    for _, trade in trades_df.iterrows():
        # Calculate bet size
        if bankroll_pct is not None:
            effective_base = bankroll * (bankroll_pct / 100)
        else:
            effective_base = base_bet

        # Apply multiplier based on strategy
        if strategy == 'flat':
            bet_size = effective_base
        elif strategy == 'mart1':
            bet_size = effective_base * current_mult
        elif strategy == 'mart3':
            bet_size = effective_base * current_mult
        elif strategy == 'mart_custom':
            bet_size = effective_base * current_mult
        elif strategy == 'infinite_mart':
            bet_size = effective_base * current_mult
        else:
            bet_size = effective_base

        # Cap bet at available bankroll
        bet_size = min(bet_size, bankroll)

        if bet_size <= 0:
            # Busted
            break

        total_wagered += bet_size

        # Use actual profit from CSV and scale by bet size
        # The CSV already has real profit with fees included
        original_bet = trade.get('Bet Size', 10)
        if original_bet == 0:
            original_bet = 10
        scale_factor = bet_size / original_bet

        original_profit = trade.get('Profit', 0)

        if trade['Outcome'] == 'win':
            net_profit = original_profit * scale_factor
            bankroll += net_profit
            consec_losses = 0
            current_mult = 1
        else:
            # Loss - scale the loss (includes fee)
            net_profit = original_profit * scale_factor  # This is negative
            bankroll += net_profit  # Adding negative = subtracting
            consec_losses += 1

            # Update multiplier based on strategy
            if strategy == 'mart1':
                current_mult = min(current_mult * 2, max_multiplier)
            elif strategy == 'mart3':
                if consec_losses >= 3:
                    current_mult = min(current_mult * 2, max_multiplier)
            elif strategy == 'mart_custom':
                if consec_losses >= mart_start_after:
                    current_mult = min(current_mult * 2, max_multiplier)
            elif strategy == 'infinite_mart':
                current_mult = current_mult * 2  # No cap

        # Track drawdown
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        current_drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        max_drawdown = max(max_drawdown, current_drawdown)

        results.append({
            'timestamp': trade.get('Timestamp', ''),
            'window': trade.get('Window', ''),
            'direction': trade.get('Direction', ''),
            'entry_price': trade['Entry Price'],
            'outcome': trade['Outcome'],
            'bet_size': bet_size,
            'profit': net_profit,
            'bankroll': bankroll,
            'multiplier': current_mult if strategy != 'flat' else 1
        })

    return {
        'trades': results,
        'final_bankroll': bankroll,
        'total_profit': bankroll - starting_bankroll,
        'total_wagered': total_wagered,
        'roi': ((bankroll - starting_bankroll) / total_wagered * 100) if total_wagered > 0 else 0,
        'max_drawdown': max_drawdown * 100,
        'num_trades': len(results),
        'wins': sum(1 for r in results if r['outcome'] == 'win'),
        'losses': sum(1 for r in results if r['outcome'] == 'loss'),
        'busted': bankroll <= 0
    }

def analyze_all_strategies(df, settings):
    """Run all strategy simulations for a given set of trades"""
    strategies = {
        'flat': 'Flat Betting',
        'mart1': 'Martingale (Every Loss)',
        'mart3': 'Martingale (After 3)',
        'mart_custom': f'Martingale (After {settings.get("mart_start_after", 1)})'
    }

    if settings.get('include_infinite'):
        strategies['infinite_mart'] = 'Infinite Martingale'

    results = {}
    for strat_key, strat_name in strategies.items():
        result = simulate_strategy(
            df,
            strategy=strat_key,
            base_bet=settings.get('base_bet', 10),
            starting_bankroll=settings.get('starting_bankroll', 1000),
            bankroll_pct=settings.get('bankroll_pct'),
            mart_start_after=settings.get('mart_start_after', 1),
            max_multiplier=settings.get('max_multiplier', 64)
        )
        result['name'] = strat_name
        results[strat_key] = result

    return results

def get_bots_from_df(df):
    """Extract unique bot IDs from dataframe"""
    if 'Bot ID' in df.columns:
        return sorted(df['Bot ID'].unique())
    return []

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    try:
        # Generate unique ID and save to disk
        file_id = str(uuid.uuid4())
        filepath = get_upload_path(file_id)
        file.save(filepath)

        # Verify it's valid CSV
        df = pd.read_csv(filepath)

        # Store only the file ID in session (small!)
        session['file_id'] = file_id
        session['filename'] = file.filename

        flash(f'Loaded {len(df)} trades from {file.filename}', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f'Error loading CSV: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    df = load_trades_df()
    if df is None:
        flash('Please upload a CSV first', 'error')
        return redirect(url_for('index'))
    bots = get_bots_from_df(df)

    # Get settings from query params or defaults
    settings = {
        'max_entry_price': int(request.args.get('max_entry', 90)),
        'base_bet': float(request.args.get('base_bet', 10)),
        'starting_bankroll': float(request.args.get('bankroll', 10000)),
        'bankroll_pct': float(request.args.get('bankroll_pct')) if request.args.get('bankroll_pct') else None,
        'mart_start_after': int(request.args.get('mart_after', 3)),
        'max_multiplier': int(request.args.get('max_mult', 64)),
        'include_infinite': request.args.get('infinite') == '1',
        'strategy': request.args.get('strategy', 'flat')
    }

    # Filter trades
    filtered_df = filter_unprofitable_trades(df, settings['max_entry_price'])

    # Get overall stats
    total_trades = len(df)
    filtered_trades = len(filtered_df)
    removed_trades = total_trades - filtered_trades

    # Run all strategy simulations on all trades combined
    all_strategies = analyze_all_strategies(filtered_df, settings)

    # Per-bot analysis for selected strategy
    bot_results = {}
    for bot in bots:
        bot_df = filtered_df[filtered_df['Bot ID'] == bot].copy()
        if len(bot_df) > 0:
            result = simulate_strategy(
                bot_df,
                strategy=settings['strategy'],
                base_bet=settings['base_bet'],
                starting_bankroll=settings['starting_bankroll'],
                bankroll_pct=settings['bankroll_pct'],
                mart_start_after=settings['mart_start_after'],
                max_multiplier=settings['max_multiplier']
            )
            bot_results[bot] = result

    # Sort bots by profit
    sorted_bots = sorted(bot_results.items(), key=lambda x: x[1]['total_profit'], reverse=True)

    return render_template('dashboard.html',
                          settings=settings,
                          total_trades=total_trades,
                          filtered_trades=filtered_trades,
                          removed_trades=removed_trades,
                          all_strategies=all_strategies,
                          bot_results=sorted_bots,
                          bots=bots,
                          filename=session.get('filename', 'Unknown'))

@app.route('/bot/<bot_id>')
@login_required
def bot_detail(bot_id):
    df = load_trades_df()
    if df is None:
        flash('Please upload a CSV first', 'error')
        return redirect(url_for('index'))

    # Get settings
    settings = {
        'max_entry_price': int(request.args.get('max_entry', 90)),
        'base_bet': float(request.args.get('base_bet', 10)),
        'starting_bankroll': float(request.args.get('bankroll', 10000)),
        'bankroll_pct': float(request.args.get('bankroll_pct')) if request.args.get('bankroll_pct') else None,
        'mart_start_after': int(request.args.get('mart_after', 3)),
        'max_multiplier': int(request.args.get('max_mult', 64)),
        'strategy': request.args.get('strategy', 'flat')
    }

    # Filter to this bot
    bot_df = df[df['Bot ID'] == bot_id].copy()
    if len(bot_df) == 0:
        flash(f'Bot {bot_id} not found', 'error')
        return redirect(url_for('dashboard'))

    # Apply entry price filter
    filtered_df = filter_unprofitable_trades(bot_df, settings['max_entry_price'])

    # Run simulation
    result = simulate_strategy(
        filtered_df,
        strategy=settings['strategy'],
        base_bet=settings['base_bet'],
        starting_bankroll=settings['starting_bankroll'],
        bankroll_pct=settings['bankroll_pct'],
        mart_start_after=settings['mart_start_after'],
        max_multiplier=settings['max_multiplier']
    )

    # All strategies for comparison
    all_strategies = {}
    for strat in ['flat', 'mart1', 'mart3']:
        all_strategies[strat] = simulate_strategy(
            filtered_df,
            strategy=strat,
            base_bet=settings['base_bet'],
            starting_bankroll=settings['starting_bankroll'],
            bankroll_pct=settings['bankroll_pct'],
            mart_start_after=settings['mart_start_after'],
            max_multiplier=settings['max_multiplier']
        )

    return render_template('bot_detail.html',
                          bot_id=bot_id,
                          settings=settings,
                          result=result,
                          all_strategies=all_strategies,
                          total_trades=len(bot_df),
                          filtered_trades=len(filtered_df))

@app.route('/download/all')
@login_required
def download_all():
    df = load_trades_df()
    if df is None:
        return "No data", 400
    max_entry = int(request.args.get('max_entry', 90))
    filtered_df = filter_unprofitable_trades(df, max_entry)

    output = io.StringIO()
    filtered_df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'filtered_trades_max{max_entry}c.csv'
    )

@app.route('/download/bot/<bot_id>')
@login_required
def download_bot(bot_id):
    df = load_trades_df()
    if df is None:
        return "No data", 400
    max_entry = int(request.args.get('max_entry', 90))

    bot_df = df[df['Bot ID'] == bot_id].copy()
    filtered_df = filter_unprofitable_trades(bot_df, max_entry)

    output = io.StringIO()
    filtered_df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'{bot_id}_filtered_max{max_entry}c.csv'
    )

@app.route('/projections')
@login_required
def projections():
    """Show weekly/monthly/yearly projections based on current data"""
    df = load_trades_df()
    if df is None:
        flash('Please upload a CSV first', 'error')
        return redirect(url_for('index'))

    settings = {
        'max_entry_price': int(request.args.get('max_entry', 90)),
        'base_bet': float(request.args.get('base_bet', 100)),
        'starting_bankroll': float(request.args.get('bankroll', 10000)),
        'strategy': request.args.get('strategy', 'mart3'),
        'mart_start_after': int(request.args.get('mart_after', 3)),
    }

    filtered_df = filter_unprofitable_trades(df, settings['max_entry_price'])

    # Calculate daily stats
    result = simulate_strategy(
        filtered_df,
        strategy=settings['strategy'],
        base_bet=settings['base_bet'],
        starting_bankroll=settings['starting_bankroll'],
        mart_start_after=settings['mart_start_after']
    )

    # Estimate time period of data (rough)
    days_of_data = 1.5  # Approximate - could parse timestamps

    daily_profit = result['total_profit'] / days_of_data
    daily_roi = (daily_profit / settings['starting_bankroll']) * 100

    projections = {
        'daily': daily_profit,
        'weekly': daily_profit * 7,
        'monthly': daily_profit * 30,
        'yearly': daily_profit * 365,
        'daily_roi': daily_roi,
        'weekly_roi': daily_roi * 7,
        'monthly_roi': daily_roi * 30,
        'yearly_roi': daily_roi * 365,
    }

    # Different bet size projections
    bet_sizes = [100, 250, 500, 1000, 2500, 5000]
    bet_projections = []
    for bet in bet_sizes:
        scale = bet / settings['base_bet']
        bet_projections.append({
            'bet_size': bet,
            'daily': daily_profit * scale,
            'weekly': daily_profit * 7 * scale,
            'monthly': daily_profit * 30 * scale,
            'yearly': daily_profit * 365 * scale,
        })

    return render_template('projections.html',
                          settings=settings,
                          result=result,
                          projections=projections,
                          bet_projections=bet_projections,
                          days_of_data=days_of_data)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
