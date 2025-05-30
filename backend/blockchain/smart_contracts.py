"""
Blockchain Integration for Arkon Financial Analyzer
Smart contracts, DeFi integrations, and tokenized rewards
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_typing import Address
import aioredis
from sqlalchemy.orm import Session

from backend.models import User, Transaction, Budget
from backend.utils.logger import get_logger
from backend.config import settings

logger = get_logger(__name__)


class ChainType(Enum):
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BSC = "bsc"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"


@dataclass
class SmartContract:
    address: str
    abi: Dict
    chain: ChainType
    name: str
    version: str


@dataclass
class DeFiPosition:
    protocol: str
    chain: ChainType
    position_type: str  # lending, staking, liquidity
    asset: str
    amount: Decimal
    apy: float
    value_usd: Decimal
    rewards_earned: Decimal
    last_updated: datetime


class BlockchainManager:
    """Manages blockchain interactions and smart contracts"""
    
    def __init__(self):
        self.web3_connections = self._initialize_connections()
        self.contracts = self._load_contracts()
        self.redis_client = None
        self.gas_oracle = GasOracle()
        self.defi_aggregator = DeFiAggregator()
        
    def _initialize_connections(self) -> Dict[ChainType, Web3]:
        """Initialize Web3 connections for multiple chains"""
        connections = {}
        
        # Ethereum Mainnet
        if settings.ETHEREUM_RPC_URL:
            w3 = Web3(Web3.HTTPProvider(settings.ETHEREUM_RPC_URL))
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            connections[ChainType.ETHEREUM] = w3
            
        # Polygon
        if settings.POLYGON_RPC_URL:
            w3 = Web3(Web3.HTTPProvider(settings.POLYGON_RPC_URL))
            connections[ChainType.POLYGON] = w3
            
        # Binance Smart Chain
        if settings.BSC_RPC_URL:
            w3 = Web3(Web3.HTTPProvider(settings.BSC_RPC_URL))
            connections[ChainType.BSC] = w3
            
        return connections
        
    def _load_contracts(self) -> Dict[str, SmartContract]:
        """Load smart contract ABIs and addresses"""
        contracts = {}
        
        # Arkon Savings Vault Contract
        contracts['savings_vault'] = SmartContract(
            address=settings.SAVINGS_VAULT_ADDRESS,
            abi=self._load_abi('SavingsVault.json'),
            chain=ChainType.POLYGON,
            name="ArkonSavingsVault",
            version="1.0.0"
        )
        
        # Arkon Reward Token Contract
        contracts['reward_token'] = SmartContract(
            address=settings.REWARD_TOKEN_ADDRESS,
            abi=self._load_abi('ArkonToken.json'),
            chain=ChainType.POLYGON,
            name="ArkonToken",
            version="1.0.0"
        )
        
        # DeFi Protocol Contracts
        contracts['aave_lending'] = SmartContract(
            address="0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
            abi=self._load_abi('AaveLendingPool.json'),
            chain=ChainType.ETHEREUM,
            name="AaveLendingPool",
            version="2.0"
        )
        
        return contracts
        
    def _load_abi(self, filename: str) -> Dict:
        """Load ABI from file"""
        try:
            with open(f'backend/blockchain/abis/{filename}', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ABI {filename}: {e}")
            return {}
            
    async def create_savings_goal(
        self,
        user_id: int,
        goal_amount: Decimal,
        target_date: datetime,
        auto_deposit: bool = True
    ) -> Dict[str, Any]:
        """Create automated savings smart contract"""
        try:
            user_wallet = await self._get_user_wallet(user_id)
            w3 = self.web3_connections[ChainType.POLYGON]
            contract = w3.eth.contract(
                address=self.contracts['savings_vault'].address,
                abi=self.contracts['savings_vault'].abi
            )
            
            # Prepare transaction
            nonce = w3.eth.get_transaction_count(user_wallet.address)
            gas_price = await self.gas_oracle.get_optimal_gas_price(ChainType.POLYGON)
            
            tx = contract.functions.createSavingsGoal(
                w3.toWei(goal_amount, 'ether'),
                int(target_date.timestamp()),
                auto_deposit
            ).buildTransaction({
                'from': user_wallet.address,
                'gas': 200000,
                'gasPrice': gas_price,
                'nonce': nonce
            })
            
            # Sign and send transaction
            signed_tx = w3.eth.account.sign_transaction(tx, user_wallet.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Parse events
            goal_created_event = contract.events.GoalCreated().processReceipt(receipt)
            
            return {
                'success': True,
                'tx_hash': tx_hash.hex(),
                'goal_id': goal_created_event[0]['args']['goalId'],
                'contract_address': self.contracts['savings_vault'].address,
                'gas_used': receipt['gasUsed']
            }
            
        except Exception as e:
            logger.error(f"Failed to create savings goal: {e}")
            return {'success': False, 'error': str(e)}
            
    async def stake_rewards(
        self,
        user_id: int,
        amount: Decimal,
        lock_period_days: int = 30
    ) -> Dict[str, Any]:
        """Stake ARKN tokens for additional rewards"""
        try:
            user_wallet = await self._get_user_wallet(user_id)
            w3 = self.web3_connections[ChainType.POLYGON]
            
            # Approve token spending
            token_contract = w3.eth.contract(
                address=self.contracts['reward_token'].address,
                abi=self.contracts['reward_token'].abi
            )
            
            staking_contract = w3.eth.contract(
                address=settings.STAKING_CONTRACT_ADDRESS,
                abi=self._load_abi('StakingRewards.json')
            )
            
            # Check allowance
            current_allowance = token_contract.functions.allowance(
                user_wallet.address,
                settings.STAKING_CONTRACT_ADDRESS
            ).call()
            
            if current_allowance < w3.toWei(amount, 'ether'):
                # Approve tokens
                approve_tx = token_contract.functions.approve(
                    settings.STAKING_CONTRACT_ADDRESS,
                    w3.toWei(amount, 'ether')
                ).buildTransaction({
                    'from': user_wallet.address,
                    'gas': 100000,
                    'gasPrice': await self.gas_oracle.get_optimal_gas_price(ChainType.POLYGON),
                    'nonce': w3.eth.get_transaction_count(user_wallet.address)
                })
                
                signed_approve = w3.eth.account.sign_transaction(approve_tx, user_wallet.private_key)
                w3.eth.send_raw_transaction(signed_approve.rawTransaction)
                w3.eth.wait_for_transaction_receipt(signed_approve.hash)
            
            # Stake tokens
            stake_tx = staking_contract.functions.stake(
                w3.toWei(amount, 'ether'),
                lock_period_days * 86400  # Convert to seconds
            ).buildTransaction({
                'from': user_wallet.address,
                'gas': 150000,
                'gasPrice': await self.gas_oracle.get_optimal_gas_price(ChainType.POLYGON),
                'nonce': w3.eth.get_transaction_count(user_wallet.address)
            })
            
            signed_stake = w3.eth.account.sign_transaction(stake_tx, user_wallet.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_stake.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Calculate APY based on lock period
            apy = self._calculate_staking_apy(lock_period_days)
            
            return {
                'success': True,
                'tx_hash': tx_hash.hex(),
                'amount_staked': str(amount),
                'lock_period_days': lock_period_days,
                'estimated_apy': apy,
                'unlock_date': (datetime.now() + timedelta(days=lock_period_days)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to stake rewards: {e}")
            return {'success': False, 'error': str(e)}
            
    async def get_defi_opportunities(
        self,
        user_id: int,
        risk_tolerance: str = "medium"
    ) -> List[Dict[str, Any]]:
        """Get personalized DeFi investment opportunities"""
        try:
            opportunities = []
            
            # Get user's risk profile
            user_profile = await self._get_user_risk_profile(user_id)
            
            # Aave Lending
            if risk_tolerance in ["low", "medium"]:
                aave_rates = await self.defi_aggregator.get_aave_rates()
                for asset, rate in aave_rates.items():
                    if rate['apy'] > 2.0:  # Filter for decent yields
                        opportunities.append({
                            'protocol': 'Aave',
                            'type': 'lending',
                            'asset': asset,
                            'apy': rate['apy'],
                            'risk_score': 2,  # Low risk
                            'min_amount': 100,
                            'chain': 'ethereum',
                            'gas_estimate': rate.get('gas_estimate', 50)
                        })
            
            # Compound Finance
            compound_rates = await self.defi_aggregator.get_compound_rates()
            for asset, rate in compound_rates.items():
                opportunities.append({
                    'protocol': 'Compound',
                    'type': 'lending',
                    'asset': asset,
                    'apy': rate['apy'],
                    'risk_score': 2,
                    'min_amount': 50,
                    'chain': 'ethereum',
                    'gas_estimate': rate.get('gas_estimate', 45)
                })
            
            # Curve Finance (Stablecoin pools)
            if risk_tolerance in ["low", "medium", "high"]:
                curve_pools = await self.defi_aggregator.get_curve_pools()
                for pool in curve_pools:
                    if pool['type'] == 'stable' or risk_tolerance != "low":
                        opportunities.append({
                            'protocol': 'Curve',
                            'type': 'liquidity',
                            'asset': pool['name'],
                            'apy': pool['apy'],
                            'risk_score': 3 if pool['type'] == 'stable' else 5,
                            'min_amount': 500,
                            'chain': pool['chain'],
                            'gas_estimate': pool.get('gas_estimate', 80)
                        })
            
            # Yearn Finance Vaults
            if risk_tolerance in ["medium", "high"]:
                yearn_vaults = await self.defi_aggregator.get_yearn_vaults()
                for vault in yearn_vaults:
                    opportunities.append({
                        'protocol': 'Yearn',
                        'type': 'vault',
                        'asset': vault['name'],
                        'apy': vault['apy'],
                        'risk_score': 4,
                        'min_amount': 1000,
                        'chain': 'ethereum',
                        'strategies': vault.get('strategies', []),
                        'gas_estimate': vault.get('gas_estimate', 100)
                    })
            
            # Sort by APY and filter by user preferences
            opportunities.sort(key=lambda x: x['apy'], reverse=True)
            
            # Add personalized recommendations
            for opp in opportunities:
                opp['recommendation_score'] = self._calculate_recommendation_score(
                    opp, user_profile, risk_tolerance
                )
                opp['estimated_monthly_return'] = (opp['apy'] / 12) * opp['min_amount'] / 100
            
            return opportunities[:20]  # Return top 20 opportunities
            
        except Exception as e:
            logger.error(f"Failed to get DeFi opportunities: {e}")
            return []
            
    async def execute_defi_investment(
        self,
        user_id: int,
        opportunity: Dict[str, Any],
        amount: Decimal
    ) -> Dict[str, Any]:
        """Execute DeFi investment based on opportunity"""
        try:
            protocol = opportunity['protocol'].lower()
            
            if protocol == 'aave':
                return await self._invest_aave(user_id, opportunity, amount)
            elif protocol == 'compound':
                return await self._invest_compound(user_id, opportunity, amount)
            elif protocol == 'curve':
                return await self._invest_curve(user_id, opportunity, amount)
            elif protocol == 'yearn':
                return await self._invest_yearn(user_id, opportunity, amount)
            else:
                return {'success': False, 'error': 'Unsupported protocol'}
                
        except Exception as e:
            logger.error(f"Failed to execute DeFi investment: {e}")
            return {'success': False, 'error': str(e)}
            
    async def create_payment_stream(
        self,
        user_id: int,
        recipient: str,
        amount_per_second: Decimal,
        duration_seconds: int
    ) -> Dict[str, Any]:
        """Create Superfluid payment stream for subscriptions"""
        try:
            user_wallet = await self._get_user_wallet(user_id)
            w3 = self.web3_connections[ChainType.POLYGON]
            
            # Superfluid contract interaction
            sf_contract = w3.eth.contract(
                address=settings.SUPERFLUID_HOST_ADDRESS,
                abi=self._load_abi('SuperfluidHost.json')
            )
            
            # Create constant flow agreement
            tx = sf_contract.functions.createFlow(
                settings.SUPERFLUID_USDC_ADDRESS,  # Super token
                recipient,
                w3.toWei(amount_per_second, 'ether'),
                '0x'  # No user data
            ).buildTransaction({
                'from': user_wallet.address,
                'gas': 300000,
                'gasPrice': await self.gas_oracle.get_optimal_gas_price(ChainType.POLYGON),
                'nonce': w3.eth.get_transaction_count(user_wallet.address)
            })
            
            signed_tx = w3.eth.account.sign_transaction(tx, user_wallet.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Store stream info
            stream_id = f"{user_wallet.address}-{recipient}-{datetime.now().timestamp()}"
            await self._store_stream_info(stream_id, {
                'user_id': user_id,
                'recipient': recipient,
                'amount_per_second': str(amount_per_second),
                'start_time': datetime.now().isoformat(),
                'end_time': (datetime.now() + timedelta(seconds=duration_seconds)).isoformat(),
                'tx_hash': tx_hash.hex()
            })
            
            return {
                'success': True,
                'stream_id': stream_id,
                'tx_hash': tx_hash.hex(),
                'total_amount': str(amount_per_second * duration_seconds),
                'end_time': (datetime.now() + timedelta(seconds=duration_seconds)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create payment stream: {e}")
            return {'success': False, 'error': str(e)}
            
    def _calculate_staking_apy(self, lock_period_days: int) -> float:
        """Calculate staking APY based on lock period"""
        base_apy = 5.0
        if lock_period_days >= 365:
            return base_apy + 10.0
        elif lock_period_days >= 180:
            return base_apy + 7.0
        elif lock_period_days >= 90:
            return base_apy + 4.0
        elif lock_period_days >= 30:
            return base_apy + 2.0
        return base_apy
        
    def _calculate_recommendation_score(
        self,
        opportunity: Dict[str, Any],
        user_profile: Dict[str, Any],
        risk_tolerance: str
    ) -> float:
        """Calculate personalized recommendation score"""
        score = 0.0
        
        # APY weight
        score += opportunity['apy'] * 0.3
        
        # Risk alignment
        risk_diff = abs(opportunity['risk_score'] - user_profile.get('risk_score', 3))
        score += (5 - risk_diff) * 2
        
        # Gas efficiency
        if opportunity['gas_estimate'] < 50:
            score += 5
        elif opportunity['gas_estimate'] < 100:
            score += 3
            
        # Protocol reputation
        trusted_protocols = ['aave', 'compound', 'curve', 'yearn']
        if opportunity['protocol'].lower() in trusted_protocols:
            score += 10
            
        return min(score, 100)


class GasOracle:
    """Gas price oracle for multiple chains"""
    
    async def get_optimal_gas_price(self, chain: ChainType) -> int:
        """Get optimal gas price for chain"""
        try:
            if chain == ChainType.ETHEREUM:
                # Use ETH Gas Station API
                return await self._get_eth_gas_price()
            elif chain == ChainType.POLYGON:
                # Use Polygon Gas Station
                return await self._get_polygon_gas_price()
            elif chain == ChainType.BSC:
                # BSC typically uses 5 gwei
                return Web3.toWei(5, 'gwei')
            else:
                # Default to 20 gwei
                return Web3.toWei(20, 'gwei')
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            return Web3.toWei(30, 'gwei')  # Fallback


class DeFiAggregator:
    """Aggregates DeFi protocol data"""
    
    async def get_aave_rates(self) -> Dict[str, Dict[str, float]]:
        """Get current Aave lending rates"""
        # In production, this would call Aave's API or smart contracts
        return {
            'USDC': {'apy': 3.5, 'utilization': 80.5},
            'DAI': {'apy': 3.2, 'utilization': 75.3},
            'USDT': {'apy': 3.8, 'utilization': 82.1},
            'ETH': {'apy': 2.1, 'utilization': 65.4}
        }
    
    async def get_compound_rates(self) -> Dict[str, Dict[str, float]]:
        """Get current Compound lending rates"""
        return {
            'USDC': {'apy': 3.3, 'utilization': 78.2},
            'DAI': {'apy': 3.0, 'utilization': 72.8},
            'ETH': {'apy': 1.9, 'utilization': 63.1}
        }
    
    async def get_curve_pools(self) -> List[Dict[str, Any]]:
        """Get Curve pool information"""
        return [
            {
                'name': '3pool',
                'type': 'stable',
                'apy': 4.2,
                'tvl': 3200000000,
                'chain': 'ethereum',
                'assets': ['DAI', 'USDC', 'USDT']
            },
            {
                'name': 'stETH',
                'type': 'eth',
                'apy': 5.8,
                'tvl': 5400000000,
                'chain': 'ethereum',
                'assets': ['ETH', 'stETH']
            }
        ]
    
    async def get_yearn_vaults(self) -> List[Dict[str, Any]]:
        """Get Yearn vault information"""
        return [
            {
                'name': 'USDC yVault',
                'apy': 8.5,
                'tvl': 450000000,
                'strategies': ['Aave', 'Compound', 'Curve'],
                'risk_score': 4
            },
            {
                'name': 'ETH yVault',
                'apy': 6.2,
                'tvl': 780000000,
                'strategies': ['Lido', 'Curve', 'Convex'],
                'risk_score': 5
            }
        ]


# Smart Contract ABIs would be stored in backend/blockchain/abis/
# Example contracts:
# - SavingsVault.sol: Automated savings with interest
# - ArkonToken.sol: ERC20 reward token
# - StakingRewards.sol: Stake ARKN for rewards
# - PaymentSplitter.sol: Split payments between wallets 