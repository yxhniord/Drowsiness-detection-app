import React, { useEffect, useState } from "react";
import { StyleSheet, View, Button, Text } from "react-native";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO, decodeJpeg } from "@tensorflow/tfjs-react-native";
import * as FileSystem from "expo-file-system";

const Home = () => {
  const [detector, setDetector] = useState();
  const [res, setRes] = useState();
  useEffect(() => {
    async function loadModel() {
      console.log("[+] Application started");
      // Wait for tensorflow module to be ready
      const tfReady = await tf.ready();
      console.log("[+] Loading custom mask detection model");
      // Replce model.json and group1-shard.bin with your own custom model
      const modelJson = require("../assets/models/model.json");
      const modelWeight = require("../assets/models/group1-shard1of1.bin");
      const detect = await tf
        .loadLayersModel(bundleResourceIO(modelJson, modelWeight))
        .catch((e) => {
          console.log("[LOADING ERROR] info:", e);
        });
      // Assign model to variable
      setDetector(detect);
      console.log("[+] Model Loaded");
    }
    loadModel();
  }, []);

  function handlePredictButtonClick() {
    if (detector) {
      predict();
    }
  }

  function predict() {
    imageUrl = require("../assets/models/_1.jpg");
    const tensor = preprocess(imageUrl);
    const prediction = model.predict(tensor);
    console.log(prediction);
  }

  function preprocess(imageUrl) {
    return null;
  }

  return (
    <View>
      <Button onPress={handlePredictButtonClick} title="Detect" />
      {res ? <Text>{res}</Text> : <Text>No result</Text>}
    </View>
  );
};

export default Home;
