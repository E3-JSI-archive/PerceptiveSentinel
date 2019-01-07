# DarkSky Grabber

## Node.js
To install Node.js follow this manual: https://nodejs.org/en/download/package-manager/

## Getting repository from Github
```
git clone https://github.com/JozefStefanInstitute/PerceptiveSentinel
cd .\Utilities\DarkSkyGrabber\
npm install
mkdir data
```

## Usage
`index.js` is used to grab daily weather data.
In `config.json` copy Dark Sky secret key.
In `coordintae.json` are coordinates of city (name of city that is created in that file ).
Example:
```
"{Grad_Jablje":{
    "lat": "46.141850",
    "longitude": "14.553334"},
{...}
}
```
Use the name of city in script:
```
place = Grad_Jablje;
```