
from finrobot.agents.trading_agent import FinRLTradingAgent
from datetime import datetime, timedelta

# Define stock list
STOCK_LIST = ['AAPL', 'MSFT', 'NVDA']

# Initialize agent
agent = FinRLTradingAgent(STOCK_LIST)

# Prepare training data (last 2 years)
end_date = datetime.now()
start_date = end_date - timedelta(days=730)
train_data = agent.prepare_data(start_date.strftime('%Y-%m-%d'), 
                              end_date.strftime('%Y-%m-%d'))

# Train the agent
trained_model = agent.train_agent(train_data)

# Paper trade (using last 30 days)
test_start = end_date - timedelta(days=30)
test_data = agent.prepare_data(test_start.strftime('%Y-%m-%d'),
                             end_date.strftime('%Y-%m-%d'))

account_value, actions = agent.paper_trade(trained_model, test_data)
print("Paper Trading Results:")
print(f"Final portfolio value: ${account_value.iloc[-1][0]:.2f}")
