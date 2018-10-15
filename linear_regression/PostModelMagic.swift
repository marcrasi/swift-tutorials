extension Linear {
  func lossThatActuallyHasGradient(for predictions: Tensor<Float>, withLabels labels: Tensor<Float>) -> Float {
    return (predictions - labels)._squared()._mean()
  }

    func lossAndGradient(for inputs: Tensor<Float>, withLabels labels: Tensor<Float>) -> (Float, Linear) {
        func composedLoss(m: Linear, i: Tensor<Float>, l: Tensor<Float>) -> Float {
            let p = m.applied(to: i)
            return m.lossThatActuallyHasGradient(for: p, withLabels: l)
        }
        let (l, (grad, _, _)) = #valueAndGradient(composedLoss)(self, inputs, labels)
        return (l, grad)
    }
}

struct SGD {
  let learningRate: Float
  func fit(_ model: inout Linear, withGradients gradients: Linear) {
    model.update(withGradients: gradients) { $0 -= learningRate * $1 }
  }
}
