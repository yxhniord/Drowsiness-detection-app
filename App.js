import React, { useEffect, useRef, useState } from "react";
import { StyleSheet, Text, View, TouchableOpacity } from "react-native";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Camera } from "expo-camera";
import * as tmImage from "@teachablemachine/image";

export default function App() {
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [cameraPermission, setCameraPermission] = useState(null);
  const cameraRef = useRef(null);

  const URL = "https://teachablemachine.withgoogle.com/models/-TBuP1naF/";
  const modelURL = URL + "model.json";
  const metadataURL = URL + "metadata.json";

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestPermissionsAsync();
      setCameraPermission(status === "granted");

      await tf.ready();

      const model = await tmImage.load(modelURL, metadataURL);


      setModel(model);
    })();
  }, []);

  const predict = async (image) => {
    if (model) {
      const prediction = await model.predict(image);
      setPredictions(prediction);
    }
  };

  const handleCameraReady = async () => {
    if (cameraRef.current && model) {
      const textureDims = {
        width: 1080,
        height: 1920,
      };
      const imageSize = {
        width: 200,
        height: 200,
      };
      const imageTensor = await tf.browser.fromPixelsAsync(cameraRef.current, 3);
      const resizedImageTensor = tf.image.resizeBilinear(imageTensor, [imageSize.height, imageSize.width]);
      await predict(resizedImageTensor);
      setTimeout(handleCameraReady, 1000);
    }
  };

  return (
    <View style={styles.container}>
      {cameraPermission && (
        <Camera
          style={styles.camera}
          type={Camera.Constants.Type.front}
          onCameraReady={handleCameraReady}
          ref={cameraRef}
        />
      )}
      <View style={styles.labelContainer}>
        {predictions.map((prediction, index) => (
          <Text key={index}>
            {prediction.className}: {prediction.probability.toFixed(2)}
          </Text>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  camera: {
    width: 200,
    height: 200,
  },
  labelContainer: {
    marginTop: 20,
  },
});
