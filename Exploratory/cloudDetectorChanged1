//Changed function for cloud detection. 
//In original version in the last line (ln25) there was: "return [B04, B03, B02].map(a => gain * a);",
// because it was returning True Color image, but I want to get NDVI. 

function index(x, y) {
	return (x - y) / (x + y);
}

function clip(a) {
  return Math.max(0, Math.min(1, a));
}

let bRatio = (B03 - 0.175) / (0.39 - 0.175);
let NGDR = index(B03, B04);
let gain = 2.5;

if (B11>0.1 && bRatio > 1) { //cloud
  return [0.0,0.0,0.0];;
}

if (B11 > 0.1 && bRatio > 0 && NGDR>0) { //cloud
  return [0.0,0.0,0.0];
}

return [(B8-B4)/(B8+B4)].map(a => gain * a); 
