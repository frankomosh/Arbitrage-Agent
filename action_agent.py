
    
import os
import time
import requests
import xgboost as xgb
import numpy as np
from dotenv import load_dotenv
from starknet_py.net import RpcProvider
from starknet_py.net.gateway_client import GatewayClient as RpcProvider
from giza.agents import AgentResult, GizaAgent
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
    
    def create_agent(agent_id: int, chain: str, contracts: dict, account_alias: str):
        
      agent = GizaAgent.from_id(
        id=agent_id,
        contracts=contracts,
        chain=chain,
        account=account_alias,
    )
      return agent

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
