import React, { useEffect, useState } from "react";
import { StyleSheet, View, Button, Text } from "react-native";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";

const Home = () => {
  const [model, setModel] = useState();
  const [res, setRes] = useState();
  useEffect(() => {
    async function loadModel() {
      console.log("[+] Application started");
      // Wait for tensorflow module to be ready
      const tfReady = await tf.ready();
      console.log("[+] Loading custom mask detection model");
      const modelJson = require("../assets/models/model.json");
      const modelWeight = require("../assets/models/group1-shard1of1.bin");
      const model = await tf
        .loadLayersModel(bundleResourceIO(modelJson, modelWeight))
        .catch((e) => {
          console.log("[LOADING ERROR] info:", e);
        });
      setModel(model);
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
