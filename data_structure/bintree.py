'''
**** 二叉树 ****
通过BinTree构建一个二叉树实例
    - 使用 list of list 存储二叉树

支持:
    - 二叉树的遍历(深度遍历:中序,前序,后序, 宽度遍历:层级序列(暂时无))
    - 通过中序和前序,或者中序和后序重建二叉树

备注:
    - 通过牛客网算法题检验,见:
        https://www.nowcoder.com/questionTerminal/c8dbc39e4c784cc69c8f263b32220165

ref.:
    - '数据结构与算法(python)',裘宗燕,机械工业出版社

python: 3.5.2
@auther: Stephenlee
@github: https://github.com/RiptideBo
2018.6.1
'''


class BinTree():
    def __init__(self,root=None,left=None,right=None):
        if root is not None:
            self.tree = [root,left,right]
        else:
            self.tree = None

    def is_empty(self):
            return self.tree is None

    def data(self):
        if self.tree:
            return self.tree[0]
        else:
            return None

    def left(self):
        if self.tree:
            return self.tree[1]
        else:
            print('tree is empty')
            return None

    def right(self):
        if self.tree:
            return self.tree[-1]
        else:
            print('tree is empty')
            return None

    def set_left(self,left):
        if self.tree:
            self.tree[1] = left

    def set_right(self,right):
        if self.tree:
            self.tree[-1] = right

    @ staticmethod
    def travelsal(tree, order='pre'):
        '''
        遍历二叉树
        :param tree: list,二叉树
        :param order: str,遍历方式
            前序: pre,
            中序: in,
            后序: post,
            层级遍历: level,从左到右
        :return: str,遍历的序列
        '''

        def _travelsal(tree,order):
            if not tree:
                return ''

            root = tree[0]

            tree_left = tree[1]
            tree_right = tree[2]

            if order == 'pre':
                return root + _travelsal(tree_left, order) + _travelsal(tree_right, order)

            elif order == 'in':
                return _travelsal(tree_left, order) + root + _travelsal(tree_right, order)

            elif order == 'post':
                return _travelsal(tree_left, order) + _travelsal(tree_right, order) + root

        seq = _travelsal(tree,order)

        return seq

    @staticmethod
    def rebuild_tree(in_seq,other_seq,order='pre'):
        '''
        根据中序和前序,或者中序和后序重建二叉树
        :param in_seq: str,中序序列
        :param other_seq: str, 其他序列
        :param order: str,其他序列的类型, 前序: pre, 后序: post
        :return: BinTree 实例, 使用实例的tree属性,获取二叉树list
        '''

        def _from_in_pre(inseq,preque):
            '''递归中序和前序重建二叉树'''
            nodes = len(preque)
            if nodes < 1:
                return None

            root = preque[0]

            if nodes == 1:
                tree = [root, None, None]
                return tree

            root_idx_in = inseq.find(root)

            left_in = inseq[:root_idx_in]
            right_in = inseq[root_idx_in + 1:]

            left_pre = preque[1:1 + root_idx_in]
            right_pre = preque[root_idx_in + 1:]

            tree_all = [root, None, None]
            tree_all[1] = _from_in_pre(left_in,left_pre)
            tree_all[2] = _from_in_pre(right_in,right_pre)

            return tree_all

        def _from_in_post(inseq, postseq):
            '''递归中序和后序重建二叉树'''
            nodes = len(inseq)
            if nodes < 1:
                return None

            root = postseq[-1]

            if nodes == 1:
                tree = [root, None, None]
                return tree

            root_idx_in = inseq.find(root)

            left_in = inseq[:root_idx_in]
            right_in = inseq[root_idx_in + 1:]

            left_post = postseq[-1 - root_idx_in:-1]
            right_post = postseq[:-1 - root_idx_in]

            tree_all = [root, None, None]
            tree_all[1] = _from_in_post(left_in, left_post)
            tree_all[2] = _from_in_post(right_in, right_post)

            return tree_all


        if order=='pre':
            tree = _from_in_pre(in_seq,other_seq)
        elif order == 'post':
            tree = _from_in_post(in_seq,other_seq)
        else:
            print('指定序列2的序列类型, 前序: pre, 后序: post')
            tree = None

        if tree:
            return BinTree(*tree)
        else:
            return BinTree()




def testing():

    pre_oder = 'abcdefgh'
    in_order = 'abcdefgh'
    post_order = 'hgfedcba'

    tree1 = BinTree.rebuild_tree(in_order, pre_oder, order='pre')
    tree2 = BinTree.rebuild_tree(in_order, post_order, order='post')

    print(tree1.tree == tree2.tree)

    print(BinTree.travelsal(tree1.tree, order='pre') == pre_oder)
    print(BinTree.travelsal(tree1.tree, order='in') == in_order)

    post = BinTree.travelsal(tree1.tree, order='post')
    print(post)

if __name__ == '__main__':
    testing()