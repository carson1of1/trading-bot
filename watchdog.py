#!/usr/bin/env python3
"""
Trading Bot Watchdog - Monitors bot health and sends alerts
Checks: process running, recent log activity, critical errors, broker connection
"""

import os
import sys
import time
import subprocess
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
BOT_PID_FILE = "logs/bot.pid"
LOG_DIR = "logs"
CHECK_INTERVAL_SECONDS = 60  # Check every minute
LOG_STALE_MINUTES = 70  # Alert if no log activity for this long (hourly cycles + buffer)
CRITICAL_PATTERNS = [
    r"CRITICAL",
    r"KILL_SWITCH.*ACTIVATED",
    r"EMERGENCY",
    r"Traceback \(most recent call last\)",
    r"FATAL",
    r"API connection failed",
    r"Authentication failed",
]

# Ignore patterns (test runs, etc)
IGNORE_PATTERNS = [
    r"unittest\.mock",
    r"test_",
    r"pytest",
]

# Email config from environment
GMAIL_ADDRESS = os.getenv("ALERT_EMAIL_FROM", os.getenv("GMAIL_ADDRESS"))
GMAIL_APP_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", os.getenv("GMAIL_APP_PASSWORD"))
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", GMAIL_ADDRESS)


def play_audio_alert(alert_type="error"):
    """Play audio alert using system commands"""
    try:
        # Try different audio methods
        if alert_type == "error":
            # Urgent beeps
            for _ in range(3):
                subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/dialog-error.oga"],
                             capture_output=True, timeout=2)
                time.sleep(0.3)
        else:
            subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/dialog-warning.oga"],
                         capture_output=True, timeout=2)
    except Exception:
        # Fallback to terminal bell
        try:
            subprocess.run(["tput", "bel"], capture_output=True)
            print("\a\a\a", end="", flush=True)
        except Exception:
            print("\a", end="", flush=True)


def send_email_alert(subject, body):
    """Send email alert via Gmail"""
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print(f"[WATCHDOG] Email not configured - skipping email alert")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = ALERT_EMAIL_TO
        msg["Subject"] = f"[TRADING BOT ALERT] {subject}"

        body_with_timestamp = f"{body}\n\n---\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nHost: {os.uname().nodename}"
        msg.attach(MIMEText(body_with_timestamp, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, ALERT_EMAIL_TO, msg.as_string())

        print(f"[WATCHDOG] Email sent: {subject}")
        return True
    except Exception as e:
        print(f"[WATCHDOG] Email failed: {e}")
        return False


def check_process_running():
    """Check if bot process is running"""
    pid_path = Path(BOT_PID_FILE)
    if not pid_path.exists():
        return False, "PID file not found"

    try:
        pid = int(pid_path.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)

        # Verify it's actually python/bot.py
        result = subprocess.run(["ps", "-p", str(pid), "-o", "cmd="],
                              capture_output=True, text=True)
        if "bot.py" in result.stdout:
            return True, f"Running (PID {pid})"
        else:
            return False, f"PID {pid} exists but not bot.py"
    except ProcessLookupError:
        return False, f"Process not running (stale PID)"
    except Exception as e:
        return False, f"Error checking process: {e}"


def get_latest_log_file():
    """Get the most recent trading log file"""
    log_dir = Path(LOG_DIR)
    log_files = list(log_dir.glob("trading_*.log"))
    if not log_files:
        return None
    return max(log_files, key=lambda p: p.stat().st_mtime)


def check_log_activity():
    """Check if log file has recent activity"""
    log_file = get_latest_log_file()
    if not log_file:
        return False, "No log files found"

    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
    age = datetime.now() - mtime

    if age > timedelta(minutes=LOG_STALE_MINUTES):
        return False, f"Log stale ({int(age.total_seconds() / 60)} min old)"

    return True, f"Active ({int(age.total_seconds())}s ago)"


def check_for_critical_errors(since_minutes=5):
    """Check recent log entries for critical errors"""
    log_file = get_latest_log_file()
    if not log_file:
        return []

    errors = []
    cutoff_time = datetime.now() - timedelta(minutes=since_minutes)

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()[-200:]  # Check last 200 lines

        for line in lines:
            # Skip lines from test runs
            if any(re.search(p, line) for p in IGNORE_PATTERNS):
                continue

            # Try to extract timestamp - skip if too old
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if match:
                try:
                    log_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                    if log_time < cutoff_time:
                        continue
                except ValueError:
                    continue  # Skip lines with unparseable timestamps
            else:
                continue  # Skip lines without timestamps

            # Check for critical patterns
            for pattern in CRITICAL_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    errors.append(line.strip())
                    break
    except Exception as e:
        errors.append(f"Error reading log: {e}")

    return errors


def check_broker_connection():
    """Quick check if broker API is reachable"""
    try:
        result = subprocess.run(
            ["python3", "-c", """
import os
from dotenv import load_dotenv
load_dotenv()
from alpaca.trading.client import TradingClient
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
account = client.get_account()
print(f"OK:{account.status}:{account.portfolio_value}")
"""],
            capture_output=True, text=True, timeout=30, cwd=Path(__file__).parent
        )
        if result.returncode == 0 and "OK:" in result.stdout:
            parts = result.stdout.strip().split(":")
            return True, f"{parts[1]} (${float(parts[2]):,.2f})"
        return False, result.stderr[:100] if result.stderr else "Unknown error"
    except subprocess.TimeoutExpired:
        return False, "Broker check timed out"
    except Exception as e:
        return False, str(e)


def run_health_check():
    """Run all health checks and return status"""
    results = {}

    # Check process
    ok, msg = check_process_running()
    results["process"] = {"ok": ok, "msg": msg}

    # Check log activity
    ok, msg = check_log_activity()
    results["log_activity"] = {"ok": ok, "msg": msg}

    # Check for critical errors
    errors = check_for_critical_errors(since_minutes=5)
    results["critical_errors"] = {"ok": len(errors) == 0, "msg": f"{len(errors)} errors", "errors": errors}

    # Check broker (less frequently - every 5th check)
    results["broker"] = {"ok": True, "msg": "Skipped", "skipped": True}

    return results


def format_alert_message(results, failed_checks):
    """Format alert message for email/console"""
    lines = ["Trading Bot Health Check FAILED\n"]
    lines.append("=" * 40)

    for check_name in failed_checks:
        result = results[check_name]
        lines.append(f"\n{check_name.upper()}: {result['msg']}")
        if check_name == "critical_errors" and result.get("errors"):
            lines.append("Recent errors:")
            for err in result["errors"][:5]:
                lines.append(f"  - {err[:100]}")

    lines.append("\n" + "=" * 40)
    lines.append("\nAll checks:")
    for name, result in results.items():
        status = "OK" if result["ok"] else "FAIL"
        lines.append(f"  [{status}] {name}: {result['msg']}")

    return "\n".join(lines)


def main():
    print(f"[WATCHDOG] Starting trading bot watchdog...")
    print(f"[WATCHDOG] Check interval: {CHECK_INTERVAL_SECONDS}s")
    print(f"[WATCHDOG] Email alerts: {'Configured' if GMAIL_ADDRESS else 'Not configured'}")
    print(f"[WATCHDOG] Press Ctrl+C to stop\n")

    last_alert_time = {}
    alert_cooldown = 300  # Don't repeat same alert within 5 minutes
    check_count = 0

    while True:
        try:
            check_count += 1
            results = run_health_check()

            # Check broker every 5 checks (5 minutes)
            if check_count % 5 == 0:
                ok, msg = check_broker_connection()
                results["broker"] = {"ok": ok, "msg": msg, "skipped": False}

            # Find failed checks
            failed_checks = [name for name, r in results.items()
                          if not r["ok"] and not r.get("skipped")]

            timestamp = datetime.now().strftime("%H:%M:%S")

            if failed_checks:
                # Check cooldown
                alert_key = ",".join(sorted(failed_checks))
                last_alert = last_alert_time.get(alert_key, 0)

                if time.time() - last_alert > alert_cooldown:
                    print(f"\n[{timestamp}] ALERT: {', '.join(failed_checks)}")

                    # Play audio alert
                    play_audio_alert("error")

                    # Send email
                    subject = f"ALERT: {', '.join(failed_checks)}"
                    body = format_alert_message(results, failed_checks)
                    send_email_alert(subject, body)

                    last_alert_time[alert_key] = time.time()
                else:
                    remaining = int(alert_cooldown - (time.time() - last_alert))
                    print(f"[{timestamp}] FAIL: {', '.join(failed_checks)} (cooldown: {remaining}s)")
            else:
                # All OK
                broker_status = results["broker"]["msg"] if not results["broker"].get("skipped") else ""
                print(f"[{timestamp}] OK | Process: {results['process']['msg']} | Log: {results['log_activity']['msg']}" +
                      (f" | Broker: {broker_status}" if broker_status else ""))

            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n[WATCHDOG] Stopped by user")
            break
        except Exception as e:
            print(f"[WATCHDOG] Error: {e}")
            time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
