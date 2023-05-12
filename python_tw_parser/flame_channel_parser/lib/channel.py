from typing import List, Sequence

class Channel(Sequence):
    def __init__(self, channel_name: str, node_type: str, node_name: str):
        self._keys = []
        self.node_type = node_type
        self.node_name = node_name
        self.name = channel_name.strip()
    
    def __len__(self) -> int:
        return len(self._keys)
    
    def __getitem__(self, index: int) -> 'Key':
        return self._keys[index]
    
    def __iter__(self) -> 'Iterator[Key]':
        return iter(self._keys)
    
    def append(self, key: 'Key') -> None:
        self._keys.append(key)
    
    def path(self) -> str:
        return '/'.join(filter(None, [self.node_name, self.name]))
    
    def to_interpolator(self) -> 'Interpolator':
        return Interpolator(self)
    
    def __repr__(self) -> str:
        return f"<Channel ({self.node_type} {self.path()}) with {len(self._keys)} keys>"

'''
Note that the type annotations are optional, but they
improve readability and help with type checking
'''