import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator>
        <Tab.Screen name="Home" component={HomeScreen} />
        <Tab.Screen name="Soil" component={SoilAnalysis} />
        <Tab.Screen name="Plant ID" component={PlantScreen} />  // Similar camera for disease
        <Tab.Screen name="Market" component={MarketScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
