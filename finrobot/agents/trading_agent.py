
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
import pandas as pd
import numpy as np

class FinRLTradingAgent:
    def __init__(self, stock_list, initial_amount=1000000):
        self.stock_list = stock_list
        self.initial_amount = initial_amount
        
    def prepare_data(self, start_date, end_date):
        # Download data
        df = YahooDownloader(start_date=start_date,
                           end_date=end_date,
                           ticker_list=self.stock_list).fetch_data()
        return df
        
    def train_agent(self, train_data):
        # Environment setup
        env_kwargs = {
            "stock_dim": len(self.stock_list),
            "hmax": 100,
            "initial_amount": self.initial_amount,
            "transaction_cost_pct": 0.001,
            "state_space": 1 + 2*len(self.stock_list),
            "action_space": len(self.stock_list),
            "tech_indicator_list": ['high', 'low', 'close', 'volume'],
            "reward_scaling": 1e-4
        }
        
        e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
        
        # Initialize agent
        agent = DRLAgent(env=e_train_gym)
        
        # Train model
        model_ppo = agent.get_model("ppo")
        trained_ppo = agent.train_model(model=model_ppo, 
                                      tb_log_name='ppo',
                                      total_timesteps=50000)
        return trained_ppo
        
    def paper_trade(self, model, test_data):
        # Set up test environment
        env_kwargs = {
            "stock_dim": len(self.stock_list),
            "hmax": 100,
            "initial_amount": self.initial_amount,
            "transaction_cost_pct": 0.001,
            "state_space": 1 + 2*len(self.stock_list),
            "action_space": len(self.stock_list),
            "tech_indicator_list": ['high', 'low', 'close', 'volume'],
            "reward_scaling": 1e-4
        }
        
        e_trade_gym = StockTradingEnv(df=test_data, **env_kwargs)
        
        # Run paper trading
        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=model,
            environment=e_trade_gym
        )
        return df_account_value, df_actions
