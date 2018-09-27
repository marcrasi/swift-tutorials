extension Tensor where Scalar == Float {
  @inlinable @inline(__always)
  @differentiable(reverse, wrt: (self), adjoint: adjCustomMean)
  public func customMean() -> Scalar {
    return self.mean()
  }

  @inlinable @inline(__always)
  public func adjCustomMean(origValue: Scalar, seed: Scalar) -> Tensor {
    return seed * Tensor(ones: self.shape) / Float(self.shape.contiguousSize)
  }

  @inlinable @inline(__always)
  @differentiable(reverse, wrt: (self), adjoint: adjCustomSquared)
  public func customSquared() -> Tensor<Scalar> {
    return self.squared()
  }

  @inlinable @inline(__always)
  public func adjCustomSquared(origValue: Tensor, seed: Tensor) -> Tensor {
    return 2 * self * seed
  }
}
