from dataclasses import dataclass, field
from typing import List, Dict, Optional

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

@dataclass
class TreeNode:
    """トークンツリーのノードを表すクラス"""
    token: str
    child_nodes: Optional[List['TreeNode']] = None

class TokenTreeConverter:
    """トークン列（逆ポーランド記法）をツリー構造に変換するクラス"""
    
    # トークン名の変換テーブル（アルファベット→トークン名）
    TOKEN_MAP = [
        '0', '1', '2', 'x', 'plus', 'minus', 'multiply', 
        'division', 'mod', 'partial_sum', 'partial_sum_of_squares',
        'self_convolution', 'linear_weighted_partial_sums', 'binomial',
        'inverse_binomial_transform', 'product_of_two_consecutive_elements',
        'cassini', 'first_stirling', 'second_stirling', 'first_differences',
        'catalan', 'sum_of_divisors', 'moebius', 'hankel', 'boustrophedon'
    ]

    def __init__(self, rpn_string: str):
        """
        Args:
            rpn_string: アルファベットで表現された逆ポーランド記法のトークン列
        """
        self.tokens = rpn_string
        self.stack: List[TreeNode] = []

    @classmethod
    def _convert_to_tokens(cls, rpn_string: str) -> List[str]:
        """アルファベットの文字列をトークン名のリストに変換

        Args:
            rpn_string: アルファベットで表現されたトークン列

        Returns:
            トークン名のリスト
        """
        return [cls.TOKEN_MAP[ord(c) - ord('A')] for c in rpn_string]

    def build_tree(self) -> TreeNode:
        """トークン列からツリー構造を構築

        Returns:
            構築されたツリーの根ノード

        Raises:
            ValueError: トークン列が不正な場合
        """
        for token in self.tokens:
            self._process_token(token)

        if len(self.stack) != 1:
            raise ValueError("Invalid token sequence: Stack should contain exactly one node after processing")

        return self.stack[0]

    def _process_token(self, token: str) -> None:
        """個々のトークンを処理してスタックを更新

        Args:
            token: 処理対象のトークン
        """
        arg_count = TOKEN_ARG_COUNTS[token]

        if arg_count == 0:
            # 定数・変数の場合
            self.stack.append(TreeNode(token=token))
        else:
            # 演算子の場合、必要な数の引数を取り出してノードを作成
            if len(self.stack) < arg_count:
                raise ValueError(f"Not enough arguments for token {token}")
                
            child_nodes = [self.stack.pop() for _ in range(arg_count)]
            child_nodes.reverse()  # 引数の順序を元に戻す
            self.stack.append(TreeNode(token=token, child_nodes=child_nodes))