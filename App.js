import { StyleSheet, View, Button, Text, Platform } from "react-native";
import { Camera, CameraType } from "expo-camera";
import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import * as FaceDetector from "expo-face-detector";
import {
  bundleResourceIO,
  cameraWithTensors,
} from "@tensorflow/tfjs-react-native";
// import Home from "./screens/Home";

const TensorCamera = cameraWithTensors(Camera);

export default function App() {
  const [type, setType] = useState(Camera.Constants.Type.front);
  const cameraRef = useRef(null);
  const [model, setModel] = useState();
  const [isLoading, setIsLoading] = useState(true);
  const [drowsiness, setDrowsiness] = useState("None");
  const faceRef = useRef();

  function argMax(arr) {
    if (arr.length === 0) {
      return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
        maxIndex = i;
        max = arr[i];
      }
    }

    return maxIndex;
  }

  const labels = ["yawn", "no_yawn"];

  useEffect(() => {
    (async () => {
      await Camera.requestCameraPermissionsAsync();
      await tf.ready();
      console.log("[+] Loading custom mask detection model");
      const modelJson = require("./assets/models_new/model.json");
      const modelWeight = require("./assets/models_new/weights.bin");
      const model = await tf
        .loadLayersModel(bundleResourceIO(modelJson, modelWeight))
        .catch((e) => {
          console.log("[LOADING ERROR] info:", e);
        });
      setModel(model);
      console.log("[+] Model Loaded");
      setIsLoading(false);
    })();
  }, []);

  function handleFlip() {
    if (type === Camera.Constants.Type.front) {
      setType(Camera.Constants.Type.back);
    } else {
      setType(Camera.Constants.Type.front);
    }
  }

  const texture =
    Platform.OS == "ios"
      ? { height: 1920, width: 1080 }
      : { height: 1200, width: 1600 };

  function handleCameraStream(images) {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      if (!nextImageTensor) {
        console.log("[LOADING ERROR] info: no model or image...");
        return;
      }
      if (
        !faceRef.current ||
        !faceRef.current.bounds ||
        !faceRef.current.leftEyeOpenProbability
      ) {
        // console.log("faceRef.current:", faceRef.current);
        requestAnimationFrame(loop);
        return;
      }
      const { x, y } = faceRef.current.bounds.origin;
      const { width, height } = faceRef.current.bounds.size;
      const { leftEyeOpenProbability, rightEyeOpenProbability } =
        faceRef.current;
      // console.log("bounds: ", faceRef.current.bounds);
      // console.log(
      //   "LeftEyeOpenProbability: ",
      //   faceRef.current.leftEyeOpenProbability
      // );
      // console.log(
      //   "RightEyeOpenProbability: ",
      //   faceRef.current.rightEyeOpenProbability
      // );
      model
        .predict(nextImageTensor.reshape([-1, 224, 224, 3]))
        .data()
        .then((prediction) => {
          console.log(prediction);
          const result = labels[argMax(prediction)];
          setDrowsiness(result);
        })
        .catch((error) => {
          console.log("[LOADING ERROR] info:", error);
        });
      requestAnimationFrame(loop);
    };
    loop();
  }

  const handleFacesDetected = (faces) => {
    const face = faces.faces[0];
    if (face) {
      faceRef.current = face ?? null;
    } else {
      faceRef.current = null;
    }
  };

  return (
    <View style={styles.container}>
      {isLoading ? (
        <Text>Loading Model...</Text>
      ) : (
        <TensorCamera
          style={styles.camera}
          type={type}
          ref={cameraRef}
          onFacesDetected={handleFacesDetected}
          faceDetectorSettings={{
            mode: FaceDetector.FaceDetectorMode.fast,
            detectLandmarks: FaceDetector.FaceDetectorLandmarks.none,
            runClassifications: FaceDetector.FaceDetectorClassifications.all,
            minDetectionInterval: 100,
            tracking: true,
          }}
          cameraTextureHeight={texture.height}
          cameraTextureWidth={texture.width}
          resizeHeight={224}
          resizeWidth={224}
          resizeDepth={3}
          onReady={handleCameraStream}
          autorender={true}
          useCustomShadersToResize={false}
        />
      )}
      <Text>Drowsiness: {drowsiness}</Text>
      <Button onPress={handleFlip} title="flip" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    justifyContent: "center",
    paddingBottom: 50,
  },
  camera: {
    flex: 1,
    borderRadius: 20,
  },
});
