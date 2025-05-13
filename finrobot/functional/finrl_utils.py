
import numpy as np
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.data_processor import DataProcessor

class FinRLUtils:
    @staticmethod
    def create_env(df, tech_indicator_list, **kwargs):
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=tech_indicator_list,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False
        )
        processed = fe.preprocess_data(df)
        
        # Set up stock trading environment
        env = StockTradingEnv(df=processed, 
                             **kwargs)
        return env
    
    @staticmethod
    def train_model(env, model="ppo", **kwargs):
        agent = DRLAgent(env=env)
        model_params = agent.get_model_params(model_name=model)
        trained_model = agent.train_model(model=model, 
                                        model_params=model_params,
                                        **kwargs)
        return trained_model

    @staticmethod
    def test_model(trained_model, test_env):
        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=trained_model,
            environment=test_env)
        return df_account_value, df_actions
