extension Linear {
    func lossAndGradient(inputs: Tensor<Float>, labels: Tensor<Float>) -> (Float, Linear) {
        func composedLoss(m: Linear, i: Tensor<Float>, l: Tensor<Float>) -> Float {
            let p = m.applied(to: i)
            return m.loss(predictions: p, labels: l)
        }
        let (l, (grad, _, _)) = #valueAndGradient(composedLoss)(self, inputs, labels)
        return (l, grad)
    }
}
