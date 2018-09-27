let np = Python.import("numpy")
let urllib = Python.import("urllib")

func normalize(_ t: Tensor<Float>) -> Tensor<Float> {
    let zeroMean = t - t.mean(alongAxes: 1)
    return zeroMean / sqrt(zeroMean.squared().mean(alongAxes: 1))
}

let downloadResult = urllib.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz",
                                        "boston_housing.npz")
let dataFilename = String(downloadResult[0])!
let data = np.load(dataFilename)
let inputs = normalize(Tensor<Float>(Tensor<Double>(numpyArray: data["x"])!))
let labels = Tensor<Float>(Tensor<Double>(numpyArray: data["y"])!)

protocol Model {
    func applied(to: Tensor<Float>) -> Tensor<Float>
    func loss(predictions: Tensor<Float>, labels: Tensor<Float>) -> Float
}
