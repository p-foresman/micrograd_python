class Node:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, children=(), operation=''):
        self.value = data
        self.gradient = 0

        # internal variables used for autograd graph construction
        self._backprop = lambda: None
        self._prev = set(children)
        self._opperation = operation # the operation that produced this node

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value + other.value, (self, other), '+')

        def _backprop():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._backprop = _backprop

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value * other.value, (self, other), '*')

        def _backprop():
            self.gradient += other.value * out.gradient
            other.gradient += self.value * out.gradient
        out._backprop = _backprop

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Node(self.value**other, (self,), f'**{other}')

        def _backprop():
            self.gradient += (other * self.value**(other-1)) * out.gradient
        out._backprop = _backprop

        return out

    def relu(self):
        out = Node(0 if self.value < 0 else self.value, (self,), 'ReLU')

        def _backprop():
            self.gradient += (out.value > 0) * out.gradient
        out._backprop = _backprop

        return out
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def backprop(self):
        # topologically sort all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.gradient = 1
        for v in reversed(topo):
            v._backprop()

    def __repr__(self):
        return f"Node(value={self.value}, gradient={self.gradient})"