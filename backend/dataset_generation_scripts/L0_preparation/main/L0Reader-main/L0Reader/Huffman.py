class HuffmanNode :
    """Node of a Huffman tree

    Attributes:
        ch: Value associated to the node
        left: pointer to the node located down and left in the tree
        right: pointer to the node located down and right in the tree
    """
    def __init__(self, ch, left,   right) :
        self.ch = ch
        self.left = left
        self.right = right
 
    def __str__(self):
        return "(" + str(self.ch) + ")"
 
class S1HuffmanCode :
    """ Huffman code trees specific for Sentinel-1 FDBAQ code.

    Class defining the huffman trees specified in the Sentinel-1 documentation
    and the methods to create de tree, to extract the code map, to encode and
    to decode a bitstream.

    Attributes:
        root: (HuffmanNode) The top level node of the tree

    Methods:
        __init__: Tree creation.
        encode: codifies a bitstream using the map associated to the tree
        decode: decodes a bitstream assuming Sentinel-1 format and using the
                map associated to the tree.
    """
    def __init__(self, k):
        """ Create the binary tree for k quantisation levels."""
        self.root = self._get_code(k)

    def _get_code(self, k) :
        """ Create the binary tree for Sentinel-1 FDBAQ mode

        Creates all the nodes needed to compose the huffman tree and then
        construct the complete tree, returning the top level node.

        :parameter
            k (integer): quantisation levels
        :return
            (HuffmanNode): Top level node of the huffman tree
        """
        queue = []
        if (k == 4) or (k == 5) or (k == 7):
            for i in range(k,0,-1):
                queue.append(HuffmanNode(i-1,None,None))
        if k == 10:
            for i in range(10,2,-1):
                queue.append(HuffmanNode(i-1, None, None))
            queue.append(HuffmanNode(-1,
                                     HuffmanNode(0, None, None),
                                     HuffmanNode(1, None, None)))
        if k == 16:
            queue.append(HuffmanNode(15,None,None))
            queue.append(HuffmanNode(14,None,None))
            queue.append(HuffmanNode(-1,
                                     HuffmanNode(12,None,None),
                                     HuffmanNode(13,None,None)))
            queue.append(HuffmanNode(-1,
                                     HuffmanNode(10, None, None),
                                     HuffmanNode(11, None, None)))
            queue.append(HuffmanNode(9, None, None))
            queue.append(HuffmanNode(8, None, None))
            queue.append(HuffmanNode(7, None, None))
            queue.append(HuffmanNode(-1,
                                     HuffmanNode(5, None, None),
                                     HuffmanNode(6, None, None)))
            queue.append(HuffmanNode(-1,
                                     HuffmanNode(3, None, None),
                                     HuffmanNode(4, None, None)))
            node2_1 = HuffmanNode(-1,
                                  HuffmanNode(1, None, None),
                                  HuffmanNode(2, None, None))
            queue.append(HuffmanNode(-1,
                                     HuffmanNode(0, None, None),
                                     node2_1))
        return self._build_tree(queue)

    def _build_tree(self, node_queue) :
        """Build a huffman tree using the nodes provided in the input list.

        This function builds a tree, from bottom to top using the node list
        provided in the list. First node of the list will be the deepest node
        of the tree ( right node in the documentation)

        :parameter
            node_queue : (list[HuffmanNode]). List of nodes to construct the
                         tree, ordered from bottom (right) to top (left)
        :return:

        """
        while len(node_queue) > 1:
            node1 = node_queue.pop(0)
            node2 = node_queue.pop(0)
            node = HuffmanNode(-1, node2, node1)
            node_queue.insert(0,node)
        return node_queue.pop(0)
 
    def _get_code_map(self):
        map = {}
        self._createCodeRec(self.root, map, "")
        return map

    def _createCodeRec(self, node, map, code):
        if node.left == None and node.right == None:
            map[node.ch] = code
            return
        self._createCodeRec(node.left, map, code + '0')
        self._createCodeRec(node.right, map, code + '1')

    def encode(self, input_data) :
        s = ""
        code_map = self._get_code_map()
        for  i in range(0, len(input_data)):
            s += code_map.get(input_data[i])
        return s
 

    def decode(self, coded, block_size) :
        error = 0
        pos = 0
        data = [0] * block_size
        sign = [0] * block_size
        curr = self.root

        for b in range (0, block_size):
            if pos == len(coded):
                error = -1
                break
            sign[b] = 1 if coded[pos] == 0 else -1
            pos += 1
            while pos < len(coded):
                curr = curr.left if not coded[pos] else curr.right
                if curr.left is None and curr.right is None:
                    data[b] = curr.ch
                    curr = self.root
                    break
                pos += 1
            if pos == len(coded):
                error = -1
                break
            pos += 1
        return error, pos, data, sign

