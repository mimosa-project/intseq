from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Set, Union
from collections import deque
import program as program
import weight

# トークンごとの引数の数を定義
TOKEN_ARG_COUNTS: Dict[str, int] = {
    # 定数・変数（引数なし）
    '0': 0, '1': 0, '2': 0, 'x': 0,
    
    # 二項演算子（引数2つ）
    'plus': 2, 'minus': 2, 'multiply': 2, 
    'division': 2, 'mod': 2,
    
    # 単項演算子（引数1つ）
    'partial_sum': 1, 'partial_sum_of_squares': 1,
    'self_convolution': 1, 'linear_weighted_partial_sums': 1,
    'binomial': 1, 'inverse_binomial_transform': 1,
    'product_of_two_consecutive_elements': 1, 'cassini': 1,
    'first_stirling': 1, 'second_stirling': 1,
    'first_differences': 1, 'catalan': 1,
    'sum_of_divisors': 1, 'moebius': 1,
    'hankel': 1, 'boustrophedon': 1
}


# 数列変換操作を表す定数辞書
# 値は変換時の必要な追加長さまたは特別な文字列を示す
NUM_REDUCING_NUMERIC_SEQUENCE_LENGTH_AFTER_CALC: Dict[str, Union[int, str]] = {
    # 基本演算（値: 0）
    '0': 0,
    '1': 0,
    '2': 0,
    'x': 0,
    'plus': 0,
    'minus': 0,
    'multiply': 0,
    'division': 0,
    'mod': 0,

    # 部分和関連の演算（値: 0）
    'partial_sum': 0,
    'partial_sum_of_squares': 0,
    'self_convolution': 0,
    'linear_weighted_partial_sums': 0,

    # 二項演算関連（値: 0）
    'binomial': 0,
    'inverse_binomial_transform': 0,

    # 追加長さ1を必要とする演算
    'product_of_two_consecutive_elements': 1,
    'first_differences': 1,

    # 追加長さ2を必要とする演算
    'cassini': 2,

    # 数論関連の変換（値: 0）
    'first_stirling': 0,
    'second_stirling': 0,
    'catalan': 0,
    'sum_of_divisors': 0,
    'moebius': 0,

    # 特別な文字列値を持つ演算
    'hankel': 'Hankel',

    # その他の変換（値: 0）
    'boustrophedon': 0
}

class Node:
    def __init__(self, token, child_nodes:Node = None):
        self.token = token
        self.child_nodes = child_nodes if child_nodes is not None else []
    
    def set_token(self, token: str):
        assert token in TOKEN_ARG_COUNTS.keys()
        self.token = token

    def append_child_node(self, node: Node):
        self.child_nodes.append(node)
    
    def get_child_node(self, i: int):
        return self.child_nodes[i]

    def generate_program(self):
        child_tokens = []
        for child_node in self.child_nodes:
            child_tokens += child_node.generate_program()
        return child_tokens + [self.token]
    
    def check_is_x_bounded(self):
        is_x_bounded = False

        if any(child.check_is_x_bounded() for child in self.child_nodes):
            is_x_bounded = True
        
        elif self.token == 'x':
            is_x_bounded = True        
        
        return is_x_bounded


# コードレビューここから(再)
class ProgramGenerator:
    
    def __init__(self, max_depth:int):
        self.max_depth : int = max_depth
        self.information_amount : int = 0
        self.root_node : Node = None
    
    def select_randam_tokens(self, num:int, is_leaf=False):
        '''
        引数の制限が一致しているトークンの中から
        weight.WEIGHTS.get_Token_weights()に応じた確率でnum個のトークンを返す
        '''
        if is_leaf:
            return random.choices(weight.WEIGHTS.get_leaf_tokens(), weights=weight.WEIGHTS.get_leaf_weights(), k=num)
        else:
            return random.choices(weight.WEIGHTS.get_tokens(), weights=weight.WEIGHTS.get_weights(), k=num)
    
    def build_tree(self, node=None, depth=0):
        if node is None:
            # root node
            token = self.select_randam_tokens(1)
            self.root_node = Node(token[0])
            self.add_information_amount(math.log(weight.WEIGHTS.get_weight_sum()/weight.WEIGHTS.get_Token_weights()[token[0]], 2))
            self.build_tree(self.root_node, depth+1)
            
        else:
            assert depth <= self.max_depth
            arg_num = TOKEN_ARG_COUNTS[node.token]
            if arg_num == 0:
                return
            
            is_leaf = (depth == self.max_depth)
            tokens = self.select_randam_tokens(arg_num, is_leaf=is_leaf)
            weight_sum = weight.WEIGHTS.get_leaf_weight_sum() if is_leaf else weight.WEIGHTS.get_weight_sum()
            for token in tokens:
                self.add_information_amount(math.log(weight_sum/weight.WEIGHTS.get_Token_weights()[token], 2))
                child_node = Node(token)
                node.append_child_node(child_node)
                if not is_leaf:
                    self.build_tree(child_node, depth+1)
    # トークン名の変換テーブル（アルファベット→トークン名）
    TOKEN_MAP = [
        '0', '1', '2', 'x', 'plus', 'minus', 'multiply', 
        'division', 'mod', 'partial_sum', 'partial_sum_of_squares',
        'self_convolution', 'linear_weighted_partial_sums', 'binomial',
        'inverse_binomial_transform', 'product_of_two_consecutive_elements',
        'cassini', 'first_stirling', 'second_stirling', 'first_differences',
        'catalan', 'sum_of_divisors', 'moebius', 'hankel', 'boustrophedon'
    ]

    @classmethod
    def _convert_to_tokens(cls, rpn_string: str) -> List[str]:
        """アルファベットの文字列をトークン名のリストに変換

        Args:
            rpn_string: アルファベットで表現されたトークン列

        Returns:
            トークン名のリスト
        """
        return [cls.TOKEN_MAP[ord(c) - ord('A')] for c in rpn_string]

    def convert_token_sequence_to_tree(self, token_sequence:list):
        """トークン列からツリー構造を構築

        Returns:
            構築されたツリーの根ノード

        Raises:
            ValueError: トークン列が不正な場合
        """
        token_tree_stack = []

        for token in token_sequence:
            token_tree_stack = self._process_token(token_tree_stack, token)

        if len(token_tree_stack) != 1:
            raise ValueError("Invalid token sequence: Stack should contain exactly one node after processing")

        self.root_node = token_tree_stack[0]

    def _process_token(self, token_tree_stack:list, token: str) -> None:
        """個々のトークンを処理してスタックを更新

        Args:
            token: 処理対象のトークン
        """
        arg_count = TOKEN_ARG_COUNTS[token]

        if arg_count == 0:
            # 定数・変数の場合
            token_tree_stack.append(Node(token=token))
        else:
            # 演算子の場合、必要な数の引数を取り出してノードを作成
            if len(token_tree_stack) < arg_count:
                raise ValueError(f"Not enough arguments for token {token}")
                
            child_nodes = [token_tree_stack.pop() for _ in range(arg_count)]
            child_nodes.reverse()  # 引数の順序を元に戻す
            token_tree_stack.append(Node(token=token, child_nodes=child_nodes))
        
        return token_tree_stack
    

    
    def add_information_amount(self, information_amount): # 引数の値だけinformation_amountに加算
        self.information_amount += information_amount
    
    def get_token_sequence(self):
        return self.root_node.generate_program()
    
    def get_information_amount(self):
        return self.information_amount
    
    def check_is_x_bounded(self):
        return self.root_node.check_is_x_bounded()

def calculate_original_sequence_length(
    tree_node: Node, 
    necessary_numeric_sequence_length: int
) -> int:
    """トークンツリーから、目標の数列長を得るために必要な入力数列の長さを計算する

    Args:
        tree_node (Node): トークン木の各ノード
        necessary_numeric_sequence_length (int): 必要とする数列の長さ

    Returns:
        int: necessary_numeric_sequence_lengthの長さの数列を生成するのに入力数列の長さ

    Note:
        - トークンの種類によって以下の2つの計算方法で処理:
          1. 固定長減少: 特定のトークンで数列長が一定数減少
          2. Hankel変換: 数列長が約1/2になる
        - 2つの入力数列を使用する場合は、短い方の長さに合わせられる
    """
    # トークンに応じた長さの調整
    if isinstance(NUM_REDUCING_NUMERIC_SEQUENCE_LENGTH_AFTER_CALC[tree_node.token], int):
        # 固定長減少の場合
        adjusted_length = necessary_numeric_sequence_length + NUM_REDUCING_NUMERIC_SEQUENCE_LENGTH_AFTER_CALC[tree_node.token]
    elif NUM_REDUCING_NUMERIC_SEQUENCE_LENGTH_AFTER_CALC[tree_node.token] == 'Hankel':
        # Hankel変換の場合
        adjusted_length = necessary_numeric_sequence_length * 2
    else:
        assert False

    # 子ノードがない場合は計算終了
    if TOKEN_ARG_COUNTS[tree_node.token] == 0:
        return adjusted_length

    # 子ノードがある場合、再帰的に計算
    max_child_length = 0
    for child_node in tree_node.child_nodes:
        child_length = calculate_original_sequence_length(child_node, adjusted_length)
        max_child_length = max(max_child_length, child_length)
    
    return max_child_length
# コードレビューここまで（再）