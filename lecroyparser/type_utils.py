from typing import Protocol

import numpy as np

class Int16Parsing(Protocol):
    def __call__(self, *, data: bytes, offset: int, position: int, endianness: str) -> np.uint16: ...

class Int32Parsing(Protocol):
    def __call__(self, *, data: bytes, offset: int, position: int, endianness: str) -> np.int32: ...

class FloatParsing(Protocol):
    def __call__(self, *, data: bytes, offset: int, position: int, endianness: str) -> np.float32: ...

class DoubleParsing(Protocol):
    def __call__(self, *, data: bytes, offset: int, position: int, endianness: str) -> np.float64: ...

class ByteParsing(Protocol):
    def __call__(self, *, data: bytes, offset: int, position: int, endianness: str) -> np.uint8: ...

class WordParsing(Protocol):
    def __call__(self, *, data: bytes, offset: int, position: int, endianness: str) -> np.int16: ...

class StringParsing(Protocol):
    def __call__(self, *, data: bytes, offset: int, position: int, endianness: str) -> bytes: ...
