import React, { useState } from 'react';
import { View, Button, Picker } from 'react-native-paper';
import * as Speech from 'expo-speech';
import { useTranslation } from 'react-i18next';  // i18n setup

const HomeScreen = () => {
  const [lang, setLang] = useState('ta');  // Tamil default

  const speakGuidance = () => {
    const msg = lang === 'ta' ? 'எங்க போக வேண்டும், என்ன செலக்ட் பண்ண வேண்டும்' : 'Kaha jana hai, kya select karna hai';
    Speech.speak(msg, { language: lang === 'ta' ? 'ta-IN' : 'hi-IN' });
  };

  return (
    <View>
      <Picker selectedValue={lang} onValueChange={setLang}>
        <Picker.Item label="தமிழ்" value="ta" />
        <Picker.Item label="हिंदी" value="hi" />
      </Picker>
      <Button onPress={speakGuidance}>Voice Help</Button>
    </View>
  );
};
