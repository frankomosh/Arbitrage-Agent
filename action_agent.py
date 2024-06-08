# import argparse
# import logging
# import os
# import pprint
# from logging import getLogger

# import numpy as np
# from dotenv import find_dotenv, load_dotenv
# from giza.agents import AgentResult, GizaAgent

# from addresses import ADDRESSES # Addresses for Nostra and ZKlend
# # from lp_tools import get_tick_range
# from uni_helpers import (fetch_zklend_data, fetch_nostra_data, approve_token, check_allowance, execute_trade,
#                          get_borrow_params)

# load_dotenv(find_dotenv())

# os.environ["DEV_PASSPHRASE"] = os.environ.get("DEV_PASSPHRASE")
# starknet_sepolia_rpc_url = os.environ.get("STARKNET_SEPOLIA_RPC_URL")

# logging.basicConfig(level=logging.INFO)

# def fetch_data(protocol: str):
#     if protocol == "zklend":
#         return fetch_zklend_data()
#     elif protocol == "nostra":
#         return fetch_nostra_data()
#     else:
#         raise ValueError("Unknown platform")
    
# def fetch_zklend_data():
#     # Implementation of logic to fetch data from ZKlend
#     realized_vol = 5.0 # Placeholder value
#     dec_price_change = 0.2 # Placeholder value
#     return realized_vol, dec_price_change

# def fetch_nostra_data():
#     # Implementation of logic to fetch data from Nostra
#     realized_vol = 4.8 # Placeholder value
#     dec_price_change = 0.5  # Placeholder Value
#     return realized_vol, dec_price_change

# def process_data(realized_vol: float, dec_price_change: float):
#     pct_change_sq = (100 * dec_price_change) ** 2
#     X = np.array([[realized_vol, pct_change_sq]])
#     return X
        
# def get_data():
#     # TODO: implement fetching onchain or from some other source
#     # hardcoding the values for now
#     realized_vol = 4.20
#     dec_price_change = 0.1
#     return realized_vol, dec_price_change


# def create_agent(
#     model_id: int, version_id: int, chain: str, contracts: dict, account: str
# ):
#     """
#     Create a Giza agent for the cross platform arbitrage
#     """
#     agent = GizaAgent(
#         contracts=contracts,
#         id=model_id,
#         version_id=version_id,
#         chain=chain,
#         account=account,
#     )
#     return agent


# def predict(agent: GizaAgent, X: np.ndarray):
#     """
#     Args:
#         X (np.ndarray): Input to the model.

#     Returns:
#         int: Predicted value.
#     """
#     prediction = agent.predict(input_feed={"val": X}, verifiable=True, job_size="XL")
#     return prediction


# def get_pred_val(prediction: AgentResult):
#     """
#     Get the value from the prediction.

#     Args:
#         prediction (dict): Prediction from the model.

#     Returns:
#         int: Predicted value.
#     """
#     # This will block the executon until the prediction has generated the proof
#     # and the proof has been verified
#     return prediction.value[0][0]

# def execute_strategy(prediction_zklend: float, prediction_nostra: float, account: str):
#     logger = getLogger("strategy_logger")
#     if prediction_zklend > prediction_nostra:
#         logger.info("Price in ZKlend is predicted to increase more than in Nostra. Borrowing from Nostra and selling in ZKlend.")
#         execute_borrow_and_sell("nostra", "zklend", account)
#     else:
#         logger.info("Prices in Nostra is predicted to increase more than the prices in ZKlend.Borrowing from ZKlend and selling in Nostra.")
#         execute_borrow_and_sell("zklend", "nostra", account)
        
# def execute_borrow_and_sell(borrow_from: str, sell_to: str, account: str):
#     # Implement the logic to borrow from one of the two protocols and sell to the other protocol
#     logger = getLogger("execution_logger")
#     logger.info(f"Executing trade: Borrow from {borrow_from} and sell to {sell_to}")
    
#     borrow_params = get_borrow_params(borrow_from, account)
#     execute_trade(borrow_from, sell_to, borrow_params, account)
        
# def get_borrow_params(protocol: str, account: str):
#     # Implement logic to get borrowing parameters from the specified protocol
#     return {
#         "amount": 1000 * 10**18, # Placeholder value
#         "collateral": 2000 * 10**18 # Placeholder value
#     }
    
# def execute_trade(borrow_from: str, sell_to: str, params: dict, account: str):
#     logger = getLogger("trade_logger")
#     # Implement the actual trade execution logic here, including the borrowing, checking allowance, approving tokens, and selling
#     logger.info(f"Borrowing from {borrow_from} and selling to {sell_to}")
#     # Placeholder for actual implementation
#     logger.info(f"Trade executed with params: {params}")

# if __name__ == "__main__":
#     # Create the parser
#     parser = argparse.ArgumentParser()

#     # Add arguments
#     parser.add_argument("--model-id", metavar="M", type=int, help="model-id")
#     parser.add_argument("--version-id", metavar="V", type=int, help="version-id")
#     parser.add_argument("--tokenA-amount", metavar="A", type=int, help="tokenA-amount")
#     parser.add_argument("--tokenB-amount", metavar="B", type=int, help="tokenB-amount")

#     # Parse arguments
#     args = parser.parse_args()

#     MODEL_ID = args.model_id
#     VERSION_ID = args.version_id
#     # tokenA_amount = args.tokenA_amount
#     # tokenB_amount = args.tokenB_amount

#     # rebalance_lp(tokenA_amount, tokenB_amount, MODEL_ID, VERSION_ID)
    
#     ACCOUNT = args.chain
    
#     logger = getLogger("main_logger")
#     logger.info("Starting the Agent")
    
#     # Fetch and process data from ZKlend
#     realized_vol_zklend, dec_price_change_zklend = fetch_data("zklend")
#     x_zklend = process_data(realized_vol_zklend, dec_price_change_zklend)
    
#     # Fetch and process data from Nostra
#     realized_vol_nostra, dec_price_change_nostra = fetch_data("nostra")
#     x_nostra = process_data(realized_vol_nostra, dec_price_change_nostra)
    
#     # Create agents for both protocols
#     contracts = {
#         "zklend": ADDRESSES["ZKLEND"],
#         "nostra": ADDRESSES["NOSTRA"]
#     }
#     agent_zklend = create_agent(MODEL_ID, VERSION_ID, CHAIN, contracts, ACCOUNT)
#     agent_nostra = create_agent(MODEL_ID, VERSION_ID, CHAIN, contracts, ACCOUNT)
    
#     # Predict using both agents
#     prediction_zklend = get_pred_val(predict(agent_zklend, x_zklend))
#     prediction_nostra = get_pred_val(predict(agent_nostra, x_nostra))
    
#     # Execute the trading strategy based on predictions
#     execute_strategy(prediction_zklend, prediction_nostra, ACCOUNT)
    
#     logger.info("Agent finished execution")
    
    
import os
import time
import requests
import xgboost as xgb
import numpy as np
from dotenv import load_dotenv
from starknet_py.net import RpcProvider
from starknet_py.net.models import Account, Call
from starknet_py.contract import Contract
from starknet_py.utils import to_hex_string

load_dotenv()

EKUBO_API_QUOTE_URL = os.getenv("EKUBO_API_QUOTE_URL")
TOKEN_TO_ARBITRAGE = os.getenv("TOKEN_TO_ARBITRAGE")
MAX_HOPS = max(2, int(os.getenv("MAX_HOPS")))
MAX_SPLITS = max(0, int(os.getenv("MAX_SPLITS")))
CHECK_INTERVAL_MS = max(3000, int(os.getenv("CHECK_INTERVAL_MS")))
MIN_POWER_OF_2 = max(32, int(os.getenv("MIN_POWER_OF_2")))
MAX_POWER_OF_2 = max(MIN_POWER_OF_2 + 1, min(65, int(os.getenv("MAX_POWER_OF_2"))))
MIN_PROFIT = max(0, int(os.getenv("MIN_PROFIT")))
NUM_TOP_QUOTES_TO_ESTIMATE = max(1, int(os.getenv("NUM_TOP_QUOTES_TO_ESTIMATE")))

JSON_RPC_URL = os.getenv("JSON_RPC_URL")

RPC_PROVIDER = RpcProvider(node_url=JSON_RPC_URL)
ACCOUNT = Account(
    provider=RPC_PROVIDER,
    address=os.getenv("ACCOUNT_ADDRESS"),
    key=os.getenv("ACCOUNT_PRIVATE_KEY")
)
ROUTER_CONTRACT = Contract(
    address=os.getenv("ROUTER_ADDRESS"),
    abi=os.getenv("ROUTER_ABI"),
    provider=RPC_PROVIDER
)

MODEL_PATH = "path/to/trained_model.xgb"

class ArbitrageAgent:
    def __init__(self, model_path):
        self.model = xgb.Booster(model_file=model_path)

    def preprocess_data(self, data):
        # Implement data preprocessing logic here
        features = []  # Extract relevant features from the raw data
        return features

    def predict_arbitrage(self, features):
        dtest = xgb.DMatrix(np.array([features]))
        prediction = self.model.predict(dtest)
        return prediction[0]

    def fetch_quote(self, amount):
        response = requests.get(f"{EKUBO_API_QUOTE_URL}/{amount}/{TOKEN_TO_ARBITRAGE}/{TOKEN_TO_ARBITRAGE}?maxHops={MAX_HOPS}&maxSplits={MAX_SPLITS}")
        if response.status_code != 200:
            return None
        return response.json()

    def execute_arbitrage(self, top_result):
        calls = top_result["calls"]
        cost = ACCOUNT.estimate_fee(calls)
        transaction_hash = ACCOUNT.execute(calls, max_fee=cost.suggested_max_fee * 2)
        receipt = RPC_PROVIDER.wait_for_transaction(transaction_hash, retry_interval=3000)
        print("Arbitrage receipt", receipt)

    def run(self):
        while True:
            top_arbitrage_results = []
            for amount in [2 ** i for i in range(MIN_POWER_OF_2, MAX_POWER_OF_2)]:
                quote = self.fetch_quote(amount)
                if not quote:
                    continue
                features = self.preprocess_data(quote)
                prediction = self.predict_arbitrage(features)
                if prediction > 0.5:
                    profit = int(quote['total']) - amount
                    if profit > MIN_PROFIT:
                        top_arbitrage_results.append({
                            "amount": amount,
                            "quote": quote,
                            "profit": profit
                        })

            if top_arbitrage_results:
                top_arbitrage_results.sort(key=lambda x: x["profit"], reverse=True)
                top_arbitrage_results = top_arbitrage_results[:NUM_TOP_QUOTES_TO_ESTIMATE]
                top_result = top_arbitrage_results[0]
                self.execute_arbitrage(top_result)
            else:
                print("No arbitrage found")

            time.sleep(CHECK_INTERVAL_MS / 1000)

if __name__ == "__main__":
    agent = ArbitrageAgent(MODEL_PATH)
    agent.run()
