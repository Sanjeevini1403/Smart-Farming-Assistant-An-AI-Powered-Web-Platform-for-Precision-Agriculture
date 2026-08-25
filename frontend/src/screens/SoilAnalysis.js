import React, { useState } from 'react';
import { TextInput, Button } from 'react-native-paper';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

const SoilAnalysis = () => {
  const [npk, setNpk] = useState({ N: '', P: '', K: '', pH: '', moisture: '' });

  const analyzePhoto = async () => {
    let result = await ImagePicker.launchCameraAsync({ base64: true });
    if (!result.canceled) {
      const formData = new FormData();
      formData.append('image', { uri: result.assets[0].uri, type: 'image/jpeg' });
      const res = await axios.post('http://your-backend/recommend_crop_photo', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      alert(`Best Crop: ${res.data.best_crop}\nFertilizer: ${res.data.fertilizer}`);
    }
  };

  const submitManual = async () => {
    const res = await axios.post('http://your-backend/recommend_crop', npk);
    alert(`Crop: ${res.data.best_crop}, Irrigate: ${res.data.irrigation}`);
  };

  return (
    <View>
      <TextInput placeholder="N" onChangeText={v => setNpk({...npk, N: v})} />
      {/* Similar for P,K,pH,moisture */}
      <Button onPress={submitManual}>Analyze</Button>
      <Button onPress={analyzePhoto}>Take Photo</Button>
    </View>
  );
};
