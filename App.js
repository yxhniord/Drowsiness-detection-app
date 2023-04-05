import { StyleSheet, View, Button, Text, Platform } from "react-native";
import { Camera, CameraType } from "expo-camera";
import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
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

  useEffect(() => {
    (async () => {
      await Camera.requestCameraPermissionsAsync();
      await tf.ready();
      console.log("[+] Loading custom mask detection model");
      const modelJson = require("./assets/models/model.json");
      const modelWeight = require("./assets/models/group1-shard1of1.bin");
      const model = await tf
        .loadLayersModel(bundleResourceIO(modelJson, modelWeight))
        .catch((e) => {
          console.log("[LOADING ERROR] info:", e);
        });
      setModel(model);
      console.log("[+] Model Loaded");
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
      if (model) {
        const nextImageTensor = images.next().value;
        if (!model || !nextImageTensor) {
          console.log("[LOADING ERROR] info: no model or image");
          return;
        }
        model
          .predict(nextImageTensor.reshape([1, 145, 145, 3]))
          .data()
          .then((prediction) => {
            console.log("[+] Predition:", prediction);
          })
          .catch((error) => {
            console.log("[LOADING ERROR] info:", error);
          });
        requestAnimationFrame(loop);
      }
    };
    loop();
  }

  return (
    <View style={styles.container}>
      <TensorCamera
        style={styles.camera}
        type={type}
        ref={cameraRef}
        cameraTextureHeight={texture.height}
        cameraTextureWidth={texture.width}
        resizeHeight={145}
        resizeWidth={145}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
        useCustomShadersToResize={false}
      />
      {/* <Camera style={styles.camera} type={type} ref={cameraRef}/> */}
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
