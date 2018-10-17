extension Tensor where Scalar == Float {
  @inlinable @inline(__always)
  @differentiable(reverse, wrt: (self), adjoint: adjCustomMean)
  public func _mean() -> Scalar {
    return self.mean()
  }

  @inlinable @inline(__always)
  public func adjCustomMean(origValue: Scalar, seed: Scalar) -> Tensor {
    return seed * Tensor(ones: self.shape) / Float(self.shape.contiguousSize)
  }

  @inlinable @inline(__always)
  @differentiable(reverse, wrt: (self), adjoint: adjCustomSquared)
  public func _squared() -> Tensor<Scalar> {
    return self.squared()
  }

  @inlinable @inline(__always)
  public func adjCustomSquared(origValue: Tensor, seed: Tensor) -> Tensor {
    return 2 * self * seed
  }
}

extension Linear {
  func lossThatActuallyHasGradient(for predictions: Tensor<Float>, labels: Tensor<Float>) -> Float {
    return (predictions - labels)._squared()._mean()
  }

    func lossAndGradients(for inputs: Tensor<Float>, labels: Tensor<Float>) -> (Float, Linear) {
        func composedLoss(m: Linear, i: Tensor<Float>, l: Tensor<Float>) -> Float {
            let p = m.applied(to: i)
            return m.lossThatActuallyHasGradient(for: p, labels: l)
        }
        let (l, (grad, _, _)) = #valueAndGradient(composedLoss)(self, inputs, labels)
        return (l, grad)
    }
}

struct SGD {
  let learningRate: Float
  func fit(_ model: inout Linear, grads: Linear) {
    model.update(withGradients: grads) { $0 -= learningRate * $1 }
  }
}
