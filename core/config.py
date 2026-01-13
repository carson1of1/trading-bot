"""
Global Configuration Manager
Loads and manages all settings from config.yaml
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
import logging
import threading


class GlobalConfig:
    """Centralized configuration management from config.yaml"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager

        Args:
            config_path: Path to config.yaml file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self._config_lock = threading.Lock()
        self.config = self._load_config()

        # Validate config on load
        warnings = self.validate_config()
        if warnings:
            self.logger.warning(f"Config validation found {len(warnings)} issues - see logs above")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file doesn't exist"""
        return {
            'mode': 'PAPER',
            'broker': {
                'paper_endpoint': 'https://paper-api.alpaca.markets',
                'live_endpoint': 'https://api.alpaca.markets'
            },
            'trading': {
                'watchlist': ['AAPL', 'NVDA', 'TSLA'],
                'timeframe': '1Hour',
                'lookback_period': 200,
            },
            'risk_management': {
                'max_position_size_pct': 3.0,
                'max_portfolio_risk_pct': 15.0,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'max_daily_loss_pct': 3.0,
                'max_open_positions': 5,
            },
            'strategies': {
                'momentum': {'enabled': True, 'weight': 0.35},
                'mean_reversion': {'enabled': True, 'weight': 0.25},
                'breakout': {'enabled': True, 'weight': 0.20},
            },
        }

    def reload(self):
        """Reload configuration from file (thread-safe)."""
        with self._config_lock:
            self.config = self._load_config()
            warnings = self.validate_config()
            if warnings:
                self.logger.warning(f"Config validation found {len(warnings)} issues after reload")
            self.logger.info("Configuration reloaded from file")

    def validate_config(self) -> List[str]:
        """Validate that config has all required keys."""
        warnings = []
        required_keys = [
            ('mode', str),
            ('risk_management', dict),
            ('trading', dict),
            ('strategies', dict),
        ]

        for key, expected_type in required_keys:
            value = self.config.get(key)
            if value is None:
                warnings.append(f"Missing required config key: '{key}'")
            elif not isinstance(value, expected_type):
                warnings.append(f"Config key '{key}' should be {expected_type.__name__}, got {type(value).__name__}")

        # Validate risk settings
        risk = self.config.get('risk_management', {})
        risk_keys = ['max_position_size_pct', 'stop_loss_pct', 'take_profit_pct', 'max_open_positions']
        for key in risk_keys:
            if key not in risk:
                warnings.append(f"Missing risk setting: '{key}'")

        # Log warnings
        for warning in warnings:
            self.logger.warning(f"Config validation: {warning}")

        return warnings

    # ========================================================================
    # GLOBAL MODE
    # ========================================================================

    def get_mode(self) -> str:
        """Get current trading mode (BACKTEST, PAPER, DRY_RUN, LIVE, TRADELOCKER)"""
        mode = self.config.get('mode', 'PAPER').upper()
        valid_modes = ['BACKTEST', 'PAPER', 'DRY_RUN', 'LIVE', 'TRADELOCKER']
        if mode not in valid_modes:
            self.logger.warning(f"Invalid mode '{mode}', defaulting to PAPER")
            return 'PAPER'
        return mode

    def set_mode(self, mode: str):
        """Set trading mode (runtime only, doesn't save to file)"""
        mode = mode.upper()
        valid_modes = ['BACKTEST', 'PAPER', 'DRY_RUN', 'LIVE', 'TRADELOCKER']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
        self.config['mode'] = mode
        self.logger.info(f"Mode changed to: {mode}")

    def is_backtest_mode(self) -> bool:
        return self.get_mode() == 'BACKTEST'

    def is_dry_run_mode(self) -> bool:
        return self.get_mode() == 'DRY_RUN'

    def is_paper_mode(self) -> bool:
        return self.get_mode() == 'PAPER'

    def is_live_mode(self) -> bool:
        return self.get_mode() == 'LIVE'

    def is_tradelocker_mode(self) -> bool:
        return self.get_mode() == 'TRADELOCKER'

    def requires_real_broker(self) -> bool:
        """Check if mode requires real broker (Alpaca API)"""
        return self.get_mode() in ['PAPER', 'LIVE']

    def requires_tradelocker_broker(self) -> bool:
        """Check if mode requires TradeLocker broker (prop firm)"""
        return self.get_mode() == 'TRADELOCKER'

    def uses_fake_broker(self) -> bool:
        """Check if mode uses fake broker (simulation)"""
        return self.get_mode() in ['BACKTEST', 'DRY_RUN']

    # ========================================================================
    # BROKER CONFIGURATION
    # ========================================================================

    def get_alpaca_endpoint(self) -> str:
        """Get appropriate Alpaca endpoint based on mode"""
        broker_config = self.config.get('broker', {})
        if self.is_live_mode():
            return broker_config.get('live_endpoint', 'https://api.alpaca.markets')
        else:
            return broker_config.get('paper_endpoint', 'https://paper-api.alpaca.markets')

    def get_fake_broker_config(self) -> Dict[str, Any]:
        """Get FakeBroker configuration"""
        return self.config.get('broker', {}).get('fake_broker', {
            'initial_cash': 100000,
            'commission_per_trade': 0,
            'slippage_percent': 0.05
        })

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., 'risk_management.max_positions')
            default: Default value if key not found
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def __repr__(self) -> str:
        return f"GlobalConfig(mode={self.get_mode()}, path={self.config_path})"


# Singleton instance
_config_instance = None
_config_lock = threading.Lock()


def get_global_config(config_path: str = "config.yaml") -> GlobalConfig:
    """Get global configuration instance (singleton, thread-safe)."""
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = GlobalConfig(config_path)
    return _config_instance
