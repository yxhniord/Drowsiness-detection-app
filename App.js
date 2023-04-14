import { StyleSheet, View, Button, Text, Platform } from "react-native";
import { Camera } from "expo-camera";
import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import * as FaceDetector from "expo-face-detector";
import {
  bundleResourceIO,
  cameraWithTensors,
} from "@tensorflow/tfjs-react-native";
import * as blazeface from "@tensorflow-models/blazeface";

const TensorCamera = cameraWithTensors(Camera);

export default function App() {
  let requestAnimationFrameId = 0;
  const texture =
    Platform.OS == "ios"
      ? { height: 1920, width: 1080 }
      : { height: 1200, width: 1600 };
  const labels = ["yawn", "no_yawn", "Closed", "Open"];

  const [type, setType] = useState(Camera.Constants.Type.front);
  const cameraRef = useRef(null);
  const [model, setModel] = useState();
  const [faceModel, setFaceModel] = useState();
  const [isLoading, setIsLoading] = useState(true);
  const [drowsiness, setDrowsiness] = useState("None");
  const faceRef = useRef();
  const [leftEyeOpenProbability, setLeftEyeOpenProbability] = useState();
  const [rightEyeOpenProbability, setRightEyeOpenProbability] = useState();

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

  useEffect(() => {
    (async () => {
      await Camera.requestCameraPermissionsAsync();
      await tf.ready();
      console.log("[+] Loading custom mask detection model");
      const modelJson = require("./assets/models/model.json");
      const modelWeight = require("./assets/models/group1-shard1of1.bin");
      const drowsymodel = await tf
        .loadLayersModel(bundleResourceIO(modelJson, modelWeight))
        .catch((e) => {
          console.log("[LOADING ERROR] info:", e);
        });
      setModel(drowsymodel);
      console.log("[+] Drowsiness Model Loaded");
      const facemodel = await blazeface.load();
      setFaceModel(facemodel);
      console.log("[+] Face Model Loaded");
      setIsLoading(false);
    })();
  }, []);

  // Run unMount for cancelling animation if it is running to avoid leaks
  useEffect(() => {
    return () => {
      cancelAnimationFrame(requestAnimationFrameId);
    };
  }, [requestAnimationFrameId]);

  function handleFlip() {
    if (type === Camera.Constants.Type.front) {
      setType(Camera.Constants.Type.back);
    } else {
      setType(Camera.Constants.Type.front);
    }
  }

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
        requestAnimationFrameId = requestAnimationFrame(loop);
        return;
      }
      // Detect eye open probability
      setLeftEyeOpenProbability(faceRef.current.leftEyeOpenProbability);
      setRightEyeOpenProbability(faceRef.current.rightEyeOpenProbability);

      // Detect face
      if (faceModel) {
        const predictions = await faceModel.estimateFaces(nextImageTensor);
        if (predictions.length > 0) {
          // You can obtain the bounding box coordinates using the below properties
          const [x1, y1] = predictions[0].topLeft;
          const [x2, y2] = predictions[0].bottomRight;

          // Calculate width and height
          const width = x2 - x1;
          const height = y2 - y1;

          // Clamp the values to ensure they stay within the bounds of the tensor
          const clampedX1 = Math.max(
            0,
            Math.min(Math.round(x1), nextImageTensor.shape[1] - 1)
          );
          const clampedY1 = Math.max(
            0,
            Math.min(Math.round(y1), nextImageTensor.shape[0] - 1)
          );
          const clampedWidth = Math.min(
            Math.round(width),
            nextImageTensor.shape[1] - clampedX1
          );
          const clampedHeight = Math.min(
            Math.round(height),
            nextImageTensor.shape[0] - clampedY1
          );
          console.log([clampedY1, clampedX1, clampedHeight, clampedWidth]);

          // If you need only the face tensor, you can crop the original tensor using the bounding box
          const faceTensor = nextImageTensor.slice(
            [clampedY1, clampedX1, 0],
            [clampedHeight, clampedWidth, 3]
          );

          const resizedFaceTensor = tf.image.resizeBilinear(
            faceTensor,
            [145, 145]
          );

          // Reshape the face tensor to [1, height, width, 3]
          const reshapedFaceTensor = resizedFaceTensor
            .expandDims(0)
            .div(tf.scalar(255));

          if (model) {
            model
              .predict(reshapedFaceTensor)
              .data()
              .then((prediction) => {
                console.log(prediction);
                const result = labels[argMax(prediction)];
                setDrowsiness(result);
              })
              .catch((error) => {
                console.log("[LOADING ERROR] info:", error);
              });
          }

          // Dispose the tensors to avoid memory leaks
          faceTensor.dispose();
          resizedFaceTensor.dispose();
          reshapedFaceTensor.dispose();
        }
      }
      requestAnimationFrameId = requestAnimationFrame(loop);
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
          resizeHeight={145}
          resizeWidth={145}
          resizeDepth={3}
          onReady={handleCameraStream}
          autorender={true}
          useCustomShadersToResize={false}
        />
      )}
      <Text>Drowsiness: {drowsiness}</Text>
      {leftEyeOpenProbability && <Text>Left: {leftEyeOpenProbability}</Text>}
      {rightEyeOpenProbability && <Text>Right: {rightEyeOpenProbability}</Text>}
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
