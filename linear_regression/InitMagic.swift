// COPIED FROM "EnableIPythonDisplay.swift"
import Python

// Workaround SR-7757.
#if canImport(Darwin)
import func Darwin.C.dlopen
#elseif canImport(Glibc)
import func Glibc.dlopen
#else
#error("Cannot import Darwin or Glibc!")
#endif
dlopen("libpython2.7.so", RTLD_NOW | RTLD_GLOBAL)

enum IPythonDisplay {
  static var socket: PythonObject = Python.None
  static var shell: PythonObject = Python.None

}

extension IPythonDisplay {
  private static func bytes(_ py: PythonObject) -> KernelCommunicator.BytesReference {
    // TODO: Replace with a faster implementation that reads bytes directly
    // from the python object's memory.
    let bytes = py.lazy.map { CChar(bitPattern: UInt8(Python.ord($0))!) }
    return KernelCommunicator.BytesReference(bytes)
  }

  private static func updateParentMessage(to parentMessage: KernelCommunicator.ParentMessage) {
    let json = Python.import("json")
    IPythonDisplay.shell.set_parent(json.loads(parentMessage.json))
  }

  private static func consumeDisplayMessages() -> [KernelCommunicator.JupyterDisplayMessage] {
    let displayMessages = IPythonDisplay.socket.messages.map {
      KernelCommunicator.JupyterDisplayMessage(parts: $0.map { bytes($0) })
    }
    IPythonDisplay.socket.messages = []
    return displayMessages
  }

  static func enable() {
    if IPythonDisplay.shell != Python.None {
      print("Warning: IPython display already enabled.")
      return
    }

    let swift_shell = Python.import("swift_shell")
    let socketAndShell = swift_shell.create_shell(
      username: JupyterKernel.communicator.jupyterSession.username,
      session_id: JupyterKernel.communicator.jupyterSession.id,
      key: JupyterKernel.communicator.jupyterSession.key)
    IPythonDisplay.socket = socketAndShell[0]
    IPythonDisplay.shell = socketAndShell[1]

    JupyterKernel.communicator.handleParentMessage(updateParentMessage)
    JupyterKernel.communicator.afterSuccessfulExecution(run: consumeDisplayMessages)
  }
}

IPythonDisplay.enable()
// END "EnableIPythonDisplay.swift"

IPythonDisplay.shell.enable_matplotlib("inline")

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
    //func applied(to: Tensor<Float>) -> Tensor<Float>
    //func loss(for predictions: Tensor<Float>, withLabels labels: Tensor<Float>) -> Float
}