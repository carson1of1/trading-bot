"""Tests for GlobalConfig module"""
import pytest
import os
import sys
import tempfile
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import GlobalConfig


class TestGlobalConfigDefaults:
    """Test default configuration values"""

    def test_default_mode_is_paper(self):
        """Default mode should be PAPER when no config file exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "nonexistent.yaml")
            cfg = GlobalConfig(config_path)
            assert cfg.get_mode() == 'PAPER'

    def test_default_has_risk_management(self):
        """Default config should have risk management settings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "nonexistent.yaml")
            cfg = GlobalConfig(config_path)
            risk = cfg.config.get('risk_management', {})
            assert 'max_position_size_pct' in risk
            assert 'stop_loss_pct' in risk
            assert 'take_profit_pct' in risk


class TestGlobalConfigLoad:
    """Test configuration loading from file"""

    def test_load_valid_config(self):
        """Should load configuration from valid YAML file"""
        config_data = {
            'mode': 'BACKTEST',
            'trading': {'watchlist': ['AAPL', 'NVDA']},
            'risk_management': {
                'max_position_size_pct': 5.0,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'max_open_positions': 3
            },
            'strategies': {'momentum': {'enabled': True}}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()

            cfg = GlobalConfig(f.name)
            assert cfg.get_mode() == 'BACKTEST'
            assert cfg.config['trading']['watchlist'] == ['AAPL', 'NVDA']

            os.unlink(f.name)

    def test_invalid_yaml_raises_error(self):
        """Should raise error for invalid YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            with pytest.raises(ValueError, match="Error parsing config"):
                GlobalConfig(f.name)

            os.unlink(f.name)


class TestModeHelpers:
    """Test mode helper methods"""

    def test_is_backtest_mode(self):
        """is_backtest_mode should return True only for BACKTEST mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({'mode': 'BACKTEST', 'trading': {}, 'risk_management': {}, 'strategies': {}}, f)

            cfg = GlobalConfig(config_path)
            assert cfg.is_backtest_mode() is True
            assert cfg.is_paper_mode() is False
            assert cfg.is_live_mode() is False

    def test_requires_real_broker(self):
        """requires_real_broker should be True for PAPER and LIVE modes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")

            # Test PAPER mode
            with open(config_path, 'w') as f:
                yaml.dump({'mode': 'PAPER', 'trading': {}, 'risk_management': {}, 'strategies': {}}, f)
            cfg = GlobalConfig(config_path)
            assert cfg.requires_real_broker() is True

            # Test BACKTEST mode
            with open(config_path, 'w') as f:
                yaml.dump({'mode': 'BACKTEST', 'trading': {}, 'risk_management': {}, 'strategies': {}}, f)
            cfg = GlobalConfig(config_path)
            assert cfg.requires_real_broker() is False

    def test_uses_fake_broker(self):
        """uses_fake_broker should be True for BACKTEST and DRY_RUN modes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")

            with open(config_path, 'w') as f:
                yaml.dump({'mode': 'BACKTEST', 'trading': {}, 'risk_management': {}, 'strategies': {}}, f)
            cfg = GlobalConfig(config_path)
            assert cfg.uses_fake_broker() is True

            with open(config_path, 'w') as f:
                yaml.dump({'mode': 'PAPER', 'trading': {}, 'risk_management': {}, 'strategies': {}}, f)
            cfg = GlobalConfig(config_path)
            assert cfg.uses_fake_broker() is False


class TestDotNotationGet:
    """Test dot notation get method"""

    def test_get_nested_value(self):
        """Should retrieve nested values using dot notation"""
        config_data = {
            'mode': 'PAPER',
            'trading': {'watchlist': ['AAPL'], 'timeframe': '1Hour'},
            'risk_management': {'max_position_size_pct': 5.0, 'stop_loss_pct': 2.0,
                               'take_profit_pct': 4.0, 'max_open_positions': 3},
            'strategies': {}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()

            cfg = GlobalConfig(f.name)
            assert cfg.get('trading.timeframe') == '1Hour'
            assert cfg.get('risk_management.max_position_size_pct') == 5.0

            os.unlink(f.name)

    def test_get_with_default(self):
        """Should return default for missing keys"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "nonexistent.yaml")
            cfg = GlobalConfig(config_path)

            assert cfg.get('nonexistent.key', 'default_value') == 'default_value'
            assert cfg.get('deeply.nested.missing.key', 42) == 42


class TestSetMode:
    """Test runtime mode changes"""

    def test_set_valid_mode(self):
        """Should allow setting valid modes at runtime"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({'mode': 'PAPER', 'trading': {}, 'risk_management': {}, 'strategies': {}}, f)

            cfg = GlobalConfig(config_path)
            cfg.set_mode('BACKTEST')
            assert cfg.get_mode() == 'BACKTEST'

    def test_set_invalid_mode_raises_error(self):
        """Should raise error for invalid mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({'mode': 'PAPER', 'trading': {}, 'risk_management': {}, 'strategies': {}}, f)

            cfg = GlobalConfig(config_path)
            with pytest.raises(ValueError, match="Invalid mode"):
                cfg.set_mode('INVALID')


class TestAlpacaEndpoint:
    """Test Alpaca endpoint selection"""

    def test_paper_endpoint_for_paper_mode(self):
        """Should return paper endpoint for PAPER mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({
                    'mode': 'PAPER',
                    'broker': {
                        'paper_endpoint': 'https://paper-api.alpaca.markets',
                        'live_endpoint': 'https://api.alpaca.markets'
                    },
                    'trading': {}, 'risk_management': {}, 'strategies': {}
                }, f)

            cfg = GlobalConfig(config_path)
            assert 'paper' in cfg.get_alpaca_endpoint()

    def test_live_endpoint_for_live_mode(self):
        """Should return live endpoint for LIVE mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({
                    'mode': 'LIVE',
                    'broker': {
                        'paper_endpoint': 'https://paper-api.alpaca.markets',
                        'live_endpoint': 'https://api.alpaca.markets'
                    },
                    'trading': {}, 'risk_management': {}, 'strategies': {}
                }, f)

            cfg = GlobalConfig(config_path)
            endpoint = cfg.get_alpaca_endpoint()
            assert 'paper' not in endpoint
            assert 'api.alpaca.markets' in endpoint


class TestDebugConfig:
    """Test debug configuration settings for signal logging"""

    def test_debug_defaults_when_missing(self):
        """Debug settings should default to False when not in config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({
                    'mode': 'PAPER',
                    'trading': {},
                    'risk_management': {},
                    'strategies': {}
                }, f)

            cfg = GlobalConfig(config_path)
            debug = cfg.get('debug', {})
            assert debug.get('log_all_signals', False) is False
            assert debug.get('log_signal_components', False) is False

    def test_debug_settings_loaded_from_config(self):
        """Debug settings should be loaded from config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({
                    'mode': 'PAPER',
                    'trading': {},
                    'risk_management': {},
                    'strategies': {},
                    'debug': {
                        'log_all_signals': True,
                        'log_signal_components': True
                    }
                }, f)

            cfg = GlobalConfig(config_path)
            debug = cfg.get('debug', {})
            assert debug.get('log_all_signals') is True
            assert debug.get('log_signal_components') is True

    def test_debug_partial_settings(self):
        """Should handle partial debug settings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({
                    'mode': 'PAPER',
                    'trading': {},
                    'risk_management': {},
                    'strategies': {},
                    'debug': {
                        'log_all_signals': True
                        # log_signal_components not specified
                    }
                }, f)

            cfg = GlobalConfig(config_path)
            debug = cfg.get('debug', {})
            assert debug.get('log_all_signals') is True
            assert debug.get('log_signal_components', False) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
