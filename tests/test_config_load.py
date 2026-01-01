"""Tests that config.yaml and universe.yaml load correctly"""
import pytest
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import GlobalConfig

# Path to the actual config files
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
UNIVERSE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'universe.yaml')


class TestConfigYamlExists:
    """Test that config.yaml exists and is valid"""

    def test_config_yaml_exists(self):
        """config.yaml should exist in repo root"""
        assert os.path.exists(CONFIG_PATH), f"config.yaml not found at {CONFIG_PATH}"

    def test_config_yaml_is_valid(self):
        """config.yaml should be valid YAML"""
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)


class TestConfigRequiredKeys:
    """Test that config.yaml has all required keys"""

    def test_has_mode(self):
        """config.yaml should have mode key"""
        cfg = GlobalConfig(CONFIG_PATH)
        mode = cfg.get_mode()
        assert mode in ['BACKTEST', 'PAPER', 'DRY_RUN', 'LIVE']

    def test_has_risk_management(self):
        """config.yaml should have risk_management section"""
        cfg = GlobalConfig(CONFIG_PATH)
        risk = cfg.config.get('risk_management')
        assert risk is not None
        assert 'max_position_size_pct' in risk
        assert 'stop_loss_pct' in risk
        assert 'take_profit_pct' in risk

    def test_has_strategies(self):
        """config.yaml should have strategies section"""
        cfg = GlobalConfig(CONFIG_PATH)
        strategies = cfg.config.get('strategies')
        assert strategies is not None
        assert 'momentum' in strategies
        assert 'mean_reversion' in strategies
        assert 'breakout' in strategies

    def test_has_trading(self):
        """config.yaml should have trading section"""
        cfg = GlobalConfig(CONFIG_PATH)
        trading = cfg.config.get('trading')
        assert trading is not None
        assert 'lookback_period' in trading

    def test_has_entry_gate(self):
        """config.yaml should have entry_gate section"""
        cfg = GlobalConfig(CONFIG_PATH)
        entry_gate = cfg.config.get('entry_gate')
        assert entry_gate is not None
        assert 'confidence_threshold' in entry_gate


class TestConfigValues:
    """Test that config values are within valid ranges"""

    def test_stop_loss_is_positive(self):
        """stop_loss_pct should be positive"""
        cfg = GlobalConfig(CONFIG_PATH)
        stop_loss = cfg.get('risk_management.stop_loss_pct')
        assert stop_loss > 0

    def test_take_profit_is_positive(self):
        """take_profit_pct should be positive"""
        cfg = GlobalConfig(CONFIG_PATH)
        take_profit = cfg.get('risk_management.take_profit_pct')
        assert take_profit > 0

    def test_confidence_threshold_in_range(self):
        """confidence_threshold should be 0-100"""
        cfg = GlobalConfig(CONFIG_PATH)
        threshold = cfg.get('entry_gate.confidence_threshold')
        assert 0 <= threshold <= 100

    def test_max_positions_reasonable(self):
        """max_open_positions should be 1-20"""
        cfg = GlobalConfig(CONFIG_PATH)
        max_pos = cfg.get('risk_management.max_open_positions')
        assert 1 <= max_pos <= 20


class TestUniverseYamlExists:
    """Test that universe.yaml exists and is valid"""

    def test_universe_yaml_exists(self):
        """universe.yaml should exist in repo root"""
        assert os.path.exists(UNIVERSE_PATH), f"universe.yaml not found at {UNIVERSE_PATH}"

    def test_universe_yaml_is_valid(self):
        """universe.yaml should be valid YAML"""
        with open(UNIVERSE_PATH, 'r') as f:
            universe = yaml.safe_load(f)
        assert isinstance(universe, dict)


class TestUniverseContent:
    """Test that universe.yaml has expected content"""

    def test_has_scanner_universe(self):
        """universe.yaml should have scanner_universe section"""
        with open(UNIVERSE_PATH, 'r') as f:
            universe = yaml.safe_load(f)
        assert 'scanner_universe' in universe
        scanner = universe['scanner_universe']
        assert isinstance(scanner, dict)
        # Should have sector categories
        assert len(scanner) > 0

    def test_has_high_volatility_symbols(self):
        """universe.yaml should have high_volatility section with key symbols"""
        with open(UNIVERSE_PATH, 'r') as f:
            universe = yaml.safe_load(f)
        high_vol = universe['scanner_universe'].get('high_volatility', [])
        # Should have our proven volatile symbols
        assert 'RGTI' in high_vol or 'IONQ' in high_vol or 'COIN' in high_vol

    def test_symbols_are_strings(self):
        """All symbols should be strings"""
        with open(UNIVERSE_PATH, 'r') as f:
            universe = yaml.safe_load(f)
        scanner = universe.get('scanner_universe', {})
        for category, symbols in scanner.items():
            if symbols:
                for sym in symbols:
                    assert isinstance(sym, str), f"Symbol {sym} in {category} is not a string"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
