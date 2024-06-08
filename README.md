# Introduction

The atomic arbitrager agent uses AI to find price differences on Starknet and executes the respective trades through the Ekubo protocol.
The primary intention of the Atomic Arbitrage Agent is to leverage price discrepancies to generate profits. The agent is meant to interact seamlessly with Starknet, using its capabilities to perform efficient, secure, and atomic transactions.
# Arbitrage in DeFi
Arbitrage involves taking advantage of price differences of the same asset across different markets to make a profit. The Atomic Arbitrage Agent focuses on identifying these discrepancies and executing transactions that exploit these price differences. This process is very dependent on the speed and efficiency of transactions, making StarkNet to be an ideal platform because of the scalability and low transaction costs.

# Setting up Dev Environment
To get started with the Atomic Arbitrage Agent, you need to set up your development environment. Ensure you have the following prerequisites:
# Pre-requisites

 1. Python 3.11 or later: Latest version of Python from the official Python website.
 2. starknet-py: Python SDK for StarkNet. Install it using `pip install starknet-py`.
 3. dotenv: The package that helps manage environment variables. To be installed using `pip install python-dotenv`.
# Installing Required Libraries
`pip install starknet-py python-dotenv`
# Environment Variables
Create a `.env` file in the root directory of the project, populated with the necessary environment variables. The variables should include addresses, URLs, and private keys needed.

# Sample .env file

ROUTER_ADDRESS="0x050d4da9f66589eadaa1d5e31cf73b08ac1a67c8b4dcd88e6fd4fe501c628af2"
EKUBO_API_QUOTE_URL="https://sepolia-api.ekubo.org/quote"
TOKEN_TO_ARBITRAGE="0x049d36570d4e46f48e99674bd3fcc84644ddd6b96f7c741b1562b82f9e004dc7"
EXPLORER_TX_PREFIX="https://starkscan.co/tx/"
JSON_RPC_URL="http://rpc.nethermind.io/sepolia-juno/"

MAX_HOPS="2"
MAX_SPLITS="3"
MIN_POWER_OF_2="53"
MAX_POWER_OF_2="65"
NUM_TOP_QUOTES_TO_ESTIMATE="5"
MIN_PROFIT="0"

CHECK_INTERVAL_MS="15000"

# Please populate with real values
ACCOUNT_PRIVATE_KEY="0x000"
ACCOUNT_ADDRESS="0x00"

# Contract ABI
The Atomic Arbitrage Agent interacts with smart contracts on Starknet. The Application Binary Interface (ABI) defines how these interactions take place. There's a JSON structure representing the ABI needed for functioning of the agent.
 ## Sample `router_abi.json`
 [
  {
    "name": "swap",
    "type": "function",
    "inputs": [
      {
        "name": "node",
        "type": "ekubo::router::SwapNode"
      },
      {
        "name": "token_amount",
        "type": "ekubo::router::TokenAmount"
      }
    ],
    "outputs": [
      {
        "name": "result",
        "type": "core::integer::u256"
      }
    ],
    "state_mutability": "view"
  }
]
## Deploying the Arbitrage Agent
Loading the ABI and Initializing the Contract
First, load the ABI from the router_abi.json file and initialize the contract using `starknet-py`.
import os
import json
from dotenv import load_dotenv
from starknet_py.net import GatewayClient
from starknet_py.contract import Contract
from starknet_py.net.models import Address

## Load environment variables from .env file
load_dotenv()

## Environment variables
ROUTER_ADDRESS = os.getenv("ROUTER_ADDRESS")
JSON_RPC_URL = os.getenv("JSON_RPC_URL")

## Load ABI from JSON file
with open("router_abi.json", "r") as abi_file:
    router_abi = json.load(abi_file)

## Initialize the client
client = GatewayClient(JSON_RPC_URL)

## Initialize the contract
router_contract = Contract(
    address=Address.from_hex(ROUTER_ADDRESS),
    abi=router_abi,
    client=client
)

## function call
async def example_call():
    # Replace with actual method and parameters
    result = await router_contract.functions["swap"].call(
        node={
            "pool_key": {
                "token0": "0x...",
                "token1": "0x...",
                "fee": 3000,
                "tick_spacing": 60,
                "extension": "0x..."
            },
            "sqrt_ratio_limit": {
                "low": 1,
                "high": 1
            },
            "skip_ahead": 1
        },
        token_amount={
            "token": "0x...",
            "amount": {
                "mag": 1000,
                "sign": 1
            }
        }
    )
    print(result)

# To run the async function
`import asyncio`
`asyncio.run(example_call())`

# Executing Arbitrage Transactions
The core functionality of the Atomic Arbitrage Agent is to execute profitable arbitrage transactions. That involves fetching quotes, calculating potential profits, and executing the swap if indeed it meets the criteria for profitability.
## import requests

def fetch_quote(token_address):
    response = requests.get(f"{os.getenv('EKUBO_API_QUOTE_URL')}?token={token_address}")
    return response.json()
# Potential Profits
def calculate_profit(quote, initial_amount):
    final_amount = quote['amount_out']
    return final_amount - initial_amount

# Giza Workflow
The Giza Platform provides tools for creating, deploying, and managing machine learning models that can be utilized within the Atomic Arbitrage Agent. Follow these steps to integrate Giza's capabilities:

## Setting up Giza
## Create a Giza User
`giza users create`
## Log into Giza

`giza users login`
# (Optional) Create an API Key

`giza users create-api-key`

## Using XGBoost Model

import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

`giza endpoints deploy --model-id 588 --version-id 2`

# Check status
`giza endpoints get-proof --endpoint-id 190 --proof-id`

# Conclusion
Through following the steps outlined, you can successfully implement an intelligent and dynamic arbitrage strategy on Starknet. Please refine and test thoroughly before deploying on a mainnet and consider factors such as gas fees, market conditions, and security measures to ensure the robustness of your arbitrage agent.

