"use strict";var DotGlobe=(()=>{var cr=Object.defineProperty,Oo=Object.getOwnPropertyDescriptor,Do=Object.getOwnPropertyNames,Ho=Object.prototype.hasOwnProperty,Qo=(t,e)=>{for(var n in e)cr(t,n,{get:e[n],enumerable:!0})},No=(t,e,n,i)=>{if(e&&typeof e=="object"||typeof e=="function")for(let r of Do(e))!Ho.call(t,r)&&r!==n&&cr(t,r,{get:()=>e[r],enumerable:!(i=Oo(e,r))||i.enumerable});return t},Go=t=>No(cr({},"__esModule",{value:!0}),t),ds={};Qo(ds,{createDotGlobe:()=>xo});var ur="160",zo=0,fs=1,jo=2,hs=1,Uo=2,Nt=3,$t=0,ut=1,Gt=2,en=0,dn=1,ps=2,gs=3,ms=4,Xo=5,fn=100,Zo=101,Fo=102,Es=103,Ss=104,Wo=200,Yo=201,Vo=202,qo=203,dr=204,fr=205,Ko=206,$o=207,eA=208,tA=209,nA=210,iA=211,rA=212,sA=213,aA=214,oA=0,AA=1,lA=2,Si=3,cA=4,uA=5,dA=6,fA=7,vs=0,hA=1,pA=2,tn=0,gA=1,mA=2,EA=3,SA=4,vA=5,xA=6,xs=300,bn=301,Bn=302,hr=303,pr=304,vi=306,gr=1e3,Bt=1001,mr=1002,dt=1003,Ms=1004,Er=1005,yt=1006,MA=1007,ii=1008,nn=1009,IA=1010,yA=1011,Sr=1012,Is=1013,rn=1014,sn=1015,ri=1016,ys=1017,Cs=1018,hn=1020,CA=1021,kt=1023,TA=1024,PA=1025,pn=1026,kn=1027,bA=1028,Ts=1029,BA=1030,Ps=1031,bs=1033,vr=33776,xr=33777,Mr=33778,Ir=33779,Bs=35840,ks=35841,Rs=35842,_s=35843,Ls=36196,ws=37492,Js=37496,Os=37808,Ds=37809,Hs=37810,Qs=37811,Ns=37812,Gs=37813,zs=37814,js=37815,Us=37816,Xs=37817,Zs=37818,Fs=37819,Ws=37820,Ys=37821,yr=36492,Vs=36494,qs=36495,kA=36283,Ks=36284,$s=36285,ea=36286,xi=2300,Mi=2301,Cr=2302,ta=2400,na=2401,ia=2402,ra=3e3,gn=3001,RA=3200,_A=3201,LA=0,wA=1,Ct="",ot="srgb",zt="srgb-linear",Tr="display-p3",Ii="display-p3-linear",yi="linear",We="srgb",Ci="rec709",Ti="p3",Rn=7680,sa=519,JA=512,OA=513,DA=514,aa=515,HA=516,QA=517,NA=518,GA=519,oa=35044,Aa="300 es",Pr=1035,jt=2e3,Pi=2001,_n=class{addEventListener(t,e){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[t]===void 0&&(n[t]=[]),n[t].indexOf(e)===-1&&n[t].push(e)}hasEventListener(t,e){if(this._listeners===void 0)return!1;const n=this._listeners;return n[t]!==void 0&&n[t].indexOf(e)!==-1}removeEventListener(t,e){if(this._listeners===void 0)return;const i=this._listeners[t];if(i!==void 0){const r=i.indexOf(e);r!==-1&&i.splice(r,1)}}dispatchEvent(t){if(this._listeners===void 0)return;const n=this._listeners[t.type];if(n!==void 0){t.target=this;const i=n.slice(0);for(let r=0,s=i.length;r<s;r++)i[r].call(this,t);t.target=null}}},At=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],la=1234567,si=Math.PI/180,ai=180/Math.PI;function Ln(){const t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,n=Math.random()*4294967295|0,i=Math.random()*4294967295|0;return(At[t&255]+At[t>>8&255]+At[t>>16&255]+At[t>>24&255]+"-"+At[e&255]+At[e>>8&255]+"-"+At[e>>16&15|64]+At[e>>24&255]+"-"+At[n&63|128]+At[n>>8&255]+"-"+At[n>>16&255]+At[n>>24&255]+At[i&255]+At[i>>8&255]+At[i>>16&255]+At[i>>24&255]).toLowerCase()}function ft(t,e,n){return Math.max(e,Math.min(n,t))}function br(t,e){return(t%e+e)%e}function zA(t,e,n,i,r){return i+(t-e)*(r-i)/(n-e)}function jA(t,e,n){return t!==e?(n-t)/(e-t):0}function oi(t,e,n){return(1-n)*t+n*e}function UA(t,e,n,i){return oi(t,e,1-Math.exp(-n*i))}function XA(t,e=1){return e-Math.abs(br(t,e*2)-e)}function ZA(t,e,n){return t<=e?0:t>=n?1:(t=(t-e)/(n-e),t*t*(3-2*t))}function FA(t,e,n){return t<=e?0:t>=n?1:(t=(t-e)/(n-e),t*t*t*(t*(t*6-15)+10))}function WA(t,e){return t+Math.floor(Math.random()*(e-t+1))}function YA(t,e){return t+Math.random()*(e-t)}function VA(t){return t*(.5-Math.random())}function qA(t){t!==void 0&&(la=t);let e=la+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}function KA(t){return t*si}function $A(t){return t*ai}function Br(t){return(t&t-1)===0&&t!==0}function el(t){return Math.pow(2,Math.ceil(Math.log(t)/Math.LN2))}function bi(t){return Math.pow(2,Math.floor(Math.log(t)/Math.LN2))}function tl(t,e,n,i,r){const s=Math.cos,o=Math.sin,a=s(n/2),A=o(n/2),l=s((e+i)/2),c=o((e+i)/2),d=s((e-i)/2),f=o((e-i)/2),p=s((i-e)/2),E=o((i-e)/2);switch(r){case"XYX":t.set(a*c,A*d,A*f,a*l);break;case"YZY":t.set(A*f,a*c,A*d,a*l);break;case"ZXZ":t.set(A*d,A*f,a*c,a*l);break;case"XZX":t.set(a*c,A*E,A*p,a*l);break;case"YXY":t.set(A*p,a*c,A*E,a*l);break;case"ZYZ":t.set(A*E,A*p,a*c,a*l);break;default:console.warn("THREE.MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+r)}}function wn(t,e){switch(e.constructor){case Float32Array:return t;case Uint32Array:return t/4294967295;case Uint16Array:return t/65535;case Uint8Array:return t/255;case Int32Array:return Math.max(t/2147483647,-1);case Int16Array:return Math.max(t/32767,-1);case Int8Array:return Math.max(t/127,-1);default:throw new Error("Invalid component type.")}}function ht(t,e){switch(e.constructor){case Float32Array:return t;case Uint32Array:return Math.round(t*4294967295);case Uint16Array:return Math.round(t*65535);case Uint8Array:return Math.round(t*255);case Int32Array:return Math.round(t*2147483647);case Int16Array:return Math.round(t*32767);case Int8Array:return Math.round(t*127);default:throw new Error("Invalid component type.")}}var kr={DEG2RAD:si,RAD2DEG:ai,generateUUID:Ln,clamp:ft,euclideanModulo:br,mapLinear:zA,inverseLerp:jA,lerp:oi,damp:UA,pingpong:XA,smoothstep:ZA,smootherstep:FA,randInt:WA,randFloat:YA,randFloatSpread:VA,seededRandom:qA,degToRad:KA,radToDeg:$A,isPowerOfTwo:Br,ceilPowerOfTwo:el,floorPowerOfTwo:bi,setQuaternionFromProperEuler:tl,normalize:ht,denormalize:wn},Fe=class Po{constructor(e=0,n=0){Po.prototype.isVector2=!0,this.x=e,this.y=n}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,n){return this.x=e,this.y=n,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const n=this.x,i=this.y,r=e.elements;return this.x=r[0]*n+r[3]*i+r[6],this.y=r[1]*n+r[4]*i+r[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,n){return this.x=Math.max(e.x,Math.min(n.x,this.x)),this.y=Math.max(e.y,Math.min(n.y,this.y)),this}clampScalar(e,n){return this.x=Math.max(e,Math.min(n,this.x)),this.y=Math.max(e,Math.min(n,this.y)),this}clampLength(e,n){const i=this.length();return this.divideScalar(i||1).multiplyScalar(Math.max(e,Math.min(n,i)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;const i=this.dot(e)/n;return Math.acos(ft(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const n=this.x-e.x,i=this.y-e.y;return n*n+i*i}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this}rotateAround(e,n){const i=Math.cos(n),r=Math.sin(n),s=this.x-e.x,o=this.y-e.y;return this.x=s*i-o*r+e.x,this.y=s*r+o*i+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}},Oe=class bo{constructor(e,n,i,r,s,o,a,A,l){bo.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,n,i,r,s,o,a,A,l)}set(e,n,i,r,s,o,a,A,l){const c=this.elements;return c[0]=e,c[1]=r,c[2]=a,c[3]=n,c[4]=s,c[5]=A,c[6]=i,c[7]=o,c[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],this}extractBasis(e,n,i){return e.setFromMatrix3Column(this,0),n.setFromMatrix3Column(this,1),i.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const n=e.elements;return this.set(n[0],n[4],n[8],n[1],n[5],n[9],n[2],n[6],n[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){const i=e.elements,r=n.elements,s=this.elements,o=i[0],a=i[3],A=i[6],l=i[1],c=i[4],d=i[7],f=i[2],p=i[5],E=i[8],g=r[0],h=r[3],u=r[6],v=r[1],y=r[4],T=r[7],_=r[2],C=r[5],B=r[8];return s[0]=o*g+a*v+A*_,s[3]=o*h+a*y+A*C,s[6]=o*u+a*T+A*B,s[1]=l*g+c*v+d*_,s[4]=l*h+c*y+d*C,s[7]=l*u+c*T+d*B,s[2]=f*g+p*v+E*_,s[5]=f*h+p*y+E*C,s[8]=f*u+p*T+E*B,this}multiplyScalar(e){const n=this.elements;return n[0]*=e,n[3]*=e,n[6]*=e,n[1]*=e,n[4]*=e,n[7]*=e,n[2]*=e,n[5]*=e,n[8]*=e,this}determinant(){const e=this.elements,n=e[0],i=e[1],r=e[2],s=e[3],o=e[4],a=e[5],A=e[6],l=e[7],c=e[8];return n*o*c-n*a*l-i*s*c+i*a*A+r*s*l-r*o*A}invert(){const e=this.elements,n=e[0],i=e[1],r=e[2],s=e[3],o=e[4],a=e[5],A=e[6],l=e[7],c=e[8],d=c*o-a*l,f=a*A-c*s,p=l*s-o*A,E=n*d+i*f+r*p;if(E===0)return this.set(0,0,0,0,0,0,0,0,0);const g=1/E;return e[0]=d*g,e[1]=(r*l-c*i)*g,e[2]=(a*i-r*o)*g,e[3]=f*g,e[4]=(c*n-r*A)*g,e[5]=(r*s-a*n)*g,e[6]=p*g,e[7]=(i*A-l*n)*g,e[8]=(o*n-i*s)*g,this}transpose(){let e;const n=this.elements;return e=n[1],n[1]=n[3],n[3]=e,e=n[2],n[2]=n[6],n[6]=e,e=n[5],n[5]=n[7],n[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const n=this.elements;return e[0]=n[0],e[1]=n[3],e[2]=n[6],e[3]=n[1],e[4]=n[4],e[5]=n[7],e[6]=n[2],e[7]=n[5],e[8]=n[8],this}setUvTransform(e,n,i,r,s,o,a){const A=Math.cos(s),l=Math.sin(s);return this.set(i*A,i*l,-i*(A*o+l*a)+o+e,-r*l,r*A,-r*(-l*o+A*a)+a+n,0,0,1),this}scale(e,n){return this.premultiply(Rr.makeScale(e,n)),this}rotate(e){return this.premultiply(Rr.makeRotation(-e)),this}translate(e,n){return this.premultiply(Rr.makeTranslation(e,n)),this}makeTranslation(e,n){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,n,0,0,1),this}makeRotation(e){const n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,i,n,0,0,0,1),this}makeScale(e,n){return this.set(e,0,0,0,n,0,0,0,1),this}equals(e){const n=this.elements,i=e.elements;for(let r=0;r<9;r++)if(n[r]!==i[r])return!1;return!0}fromArray(e,n=0){for(let i=0;i<9;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){const i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e}clone(){return new this.constructor().fromArray(this.elements)}},Rr=new Oe;function ca(t){for(let e=t.length-1;e>=0;--e)if(t[e]>=65535)return!0;return!1}function Bi(t){return document.createElementNS("http://www.w3.org/1999/xhtml",t)}function nl(){const t=Bi("canvas");return t.style.display="block",t}var ua={};function Ai(t){t in ua||(ua[t]=!0,console.warn(t))}var da=new Oe().set(.8224621,.177538,0,.0331941,.9668058,0,.0170827,.0723974,.9105199),fa=new Oe().set(1.2249401,-.2249404,0,-.0420569,1.0420571,0,-.0196376,-.0786361,1.0982735),ki={[zt]:{transfer:yi,primaries:Ci,toReference:t=>t,fromReference:t=>t},[ot]:{transfer:We,primaries:Ci,toReference:t=>t.convertSRGBToLinear(),fromReference:t=>t.convertLinearToSRGB()},[Ii]:{transfer:yi,primaries:Ti,toReference:t=>t.applyMatrix3(fa),fromReference:t=>t.applyMatrix3(da)},[Tr]:{transfer:We,primaries:Ti,toReference:t=>t.convertSRGBToLinear().applyMatrix3(fa),fromReference:t=>t.applyMatrix3(da).convertLinearToSRGB()}},il=new Set([zt,Ii]),je={enabled:!0,_workingColorSpace:zt,get workingColorSpace(){return this._workingColorSpace},set workingColorSpace(t){if(!il.has(t))throw new Error(`Unsupported working color space, "${t}".`);this._workingColorSpace=t},convert:function(t,e,n){if(this.enabled===!1||e===n||!e||!n)return t;const i=ki[e].toReference,r=ki[n].fromReference;return r(i(t))},fromWorkingColorSpace:function(t,e){return this.convert(t,this._workingColorSpace,e)},toWorkingColorSpace:function(t,e){return this.convert(t,e,this._workingColorSpace)},getPrimaries:function(t){return ki[t].primaries},getTransfer:function(t){return t===Ct?yi:ki[t].transfer}};function Jn(t){return t<.04045?t*.0773993808:Math.pow(t*.9478672986+.0521327014,2.4)}function _r(t){return t<.0031308?t*12.92:1.055*Math.pow(t,.41666)-.055}var On,ha=class{static getDataURL(t){if(/^data:/i.test(t.src)||typeof HTMLCanvasElement>"u")return t.src;let e;if(t instanceof HTMLCanvasElement)e=t;else{On===void 0&&(On=Bi("canvas")),On.width=t.width,On.height=t.height;const n=On.getContext("2d");t instanceof ImageData?n.putImageData(t,0,0):n.drawImage(t,0,0,t.width,t.height),e=On}return e.width>2048||e.height>2048?(console.warn("THREE.ImageUtils.getDataURL: Image converted to jpg for performance reasons",t),e.toDataURL("image/jpeg",.6)):e.toDataURL("image/png")}static sRGBToLinear(t){if(typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap){const e=Bi("canvas");e.width=t.width,e.height=t.height;const n=e.getContext("2d");n.drawImage(t,0,0,t.width,t.height);const i=n.getImageData(0,0,t.width,t.height),r=i.data;for(let s=0;s<r.length;s++)r[s]=Jn(r[s]/255)*255;return n.putImageData(i,0,0),e}else if(t.data){const e=t.data.slice(0);for(let n=0;n<e.length;n++)e instanceof Uint8Array||e instanceof Uint8ClampedArray?e[n]=Math.floor(Jn(e[n]/255)*255):e[n]=Jn(e[n]);return{data:e,width:t.width,height:t.height}}else return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),t}},rl=0,pa=class{constructor(t=null){this.isSource=!0,Object.defineProperty(this,"id",{value:rl++}),this.uuid=Ln(),this.data=t,this.version=0}set needsUpdate(t){t===!0&&this.version++}toJSON(t){const e=t===void 0||typeof t=="string";if(!e&&t.images[this.uuid]!==void 0)return t.images[this.uuid];const n={uuid:this.uuid,url:""},i=this.data;if(i!==null){let r;if(Array.isArray(i)){r=[];for(let s=0,o=i.length;s<o;s++)i[s].isDataTexture?r.push(Lr(i[s].image)):r.push(Lr(i[s]))}else r=Lr(i);n.url=r}return e||(t.images[this.uuid]=n),n}};function Lr(t){return typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap?ha.getDataURL(t):t.data?{data:Array.from(t.data),width:t.width,height:t.height,type:t.data.constructor.name}:(console.warn("THREE.Texture: Unable to serialize Texture."),{})}var sl=0,wt=class Ar extends _n{constructor(e=Ar.DEFAULT_IMAGE,n=Ar.DEFAULT_MAPPING,i=Bt,r=Bt,s=yt,o=ii,a=kt,A=nn,l=Ar.DEFAULT_ANISOTROPY,c=Ct){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:sl++}),this.uuid=Ln(),this.name="",this.source=new pa(e),this.mipmaps=[],this.mapping=n,this.channel=0,this.wrapS=i,this.wrapT=r,this.magFilter=s,this.minFilter=o,this.anisotropy=l,this.format=a,this.internalFormat=null,this.type=A,this.offset=new Fe(0,0),this.repeat=new Fe(1,1),this.center=new Fe(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Oe,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,typeof c=="string"?this.colorSpace=c:(Ai("THREE.Texture: Property .encoding has been replaced by .colorSpace."),this.colorSpace=c===gn?ot:Ct),this.userData={},this.version=0,this.onUpdate=null,this.isRenderTargetTexture=!1,this.needsPMREMUpdate=!1}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}toJSON(e){const n=e===void 0||typeof e=="string";if(!n&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const i={metadata:{version:4.6,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(i.userData=this.userData),n||(e.textures[this.uuid]=i),i}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==xs)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case gr:e.x=e.x-Math.floor(e.x);break;case Bt:e.x=e.x<0?0:1;break;case mr:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case gr:e.y=e.y-Math.floor(e.y);break;case Bt:e.y=e.y<0?0:1;break;case mr:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}get encoding(){return Ai("THREE.Texture: Property .encoding has been replaced by .colorSpace."),this.colorSpace===ot?gn:ra}set encoding(e){Ai("THREE.Texture: Property .encoding has been replaced by .colorSpace."),this.colorSpace=e===gn?ot:Ct}};wt.DEFAULT_IMAGE=null,wt.DEFAULT_MAPPING=xs,wt.DEFAULT_ANISOTROPY=1;var pt=class Bo{constructor(e=0,n=0,i=0,r=1){Bo.prototype.isVector4=!0,this.x=e,this.y=n,this.z=i,this.w=r}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,n,i,r){return this.x=e,this.y=n,this.z=i,this.w=r,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;case 3:this.w=n;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this.w=e.w+n.w,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this.w+=e.w*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this.w=e.w-n.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const n=this.x,i=this.y,r=this.z,s=this.w,o=e.elements;return this.x=o[0]*n+o[4]*i+o[8]*r+o[12]*s,this.y=o[1]*n+o[5]*i+o[9]*r+o[13]*s,this.z=o[2]*n+o[6]*i+o[10]*r+o[14]*s,this.w=o[3]*n+o[7]*i+o[11]*r+o[15]*s,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const n=Math.sqrt(1-e.w*e.w);return n<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/n,this.y=e.y/n,this.z=e.z/n),this}setAxisAngleFromRotationMatrix(e){let n,i,r,s;const A=e.elements,l=A[0],c=A[4],d=A[8],f=A[1],p=A[5],E=A[9],g=A[2],h=A[6],u=A[10];if(Math.abs(c-f)<.01&&Math.abs(d-g)<.01&&Math.abs(E-h)<.01){if(Math.abs(c+f)<.1&&Math.abs(d+g)<.1&&Math.abs(E+h)<.1&&Math.abs(l+p+u-3)<.1)return this.set(1,0,0,0),this;n=Math.PI;const y=(l+1)/2,T=(p+1)/2,_=(u+1)/2,C=(c+f)/4,B=(d+g)/4,z=(E+h)/4;return y>T&&y>_?y<.01?(i=0,r=.707106781,s=.707106781):(i=Math.sqrt(y),r=C/i,s=B/i):T>_?T<.01?(i=.707106781,r=0,s=.707106781):(r=Math.sqrt(T),i=C/r,s=z/r):_<.01?(i=.707106781,r=.707106781,s=0):(s=Math.sqrt(_),i=B/s,r=z/s),this.set(i,r,s,n),this}let v=Math.sqrt((h-E)*(h-E)+(d-g)*(d-g)+(f-c)*(f-c));return Math.abs(v)<.001&&(v=1),this.x=(h-E)/v,this.y=(d-g)/v,this.z=(f-c)/v,this.w=Math.acos((l+p+u-1)/2),this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,n){return this.x=Math.max(e.x,Math.min(n.x,this.x)),this.y=Math.max(e.y,Math.min(n.y,this.y)),this.z=Math.max(e.z,Math.min(n.z,this.z)),this.w=Math.max(e.w,Math.min(n.w,this.w)),this}clampScalar(e,n){return this.x=Math.max(e,Math.min(n,this.x)),this.y=Math.max(e,Math.min(n,this.y)),this.z=Math.max(e,Math.min(n,this.z)),this.w=Math.max(e,Math.min(n,this.w)),this}clampLength(e,n){const i=this.length();return this.divideScalar(i||1).multiplyScalar(Math.max(e,Math.min(n,i)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this.w+=(e.w-this.w)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this.w=e.w+(n.w-e.w)*i,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this.w=e[n+3],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e[n+3]=this.w,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this.w=e.getW(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}},al=class extends _n{constructor(t=1,e=1,n={}){super(),this.isRenderTarget=!0,this.width=t,this.height=e,this.depth=1,this.scissor=new pt(0,0,t,e),this.scissorTest=!1,this.viewport=new pt(0,0,t,e);const i={width:t,height:e,depth:1};n.encoding!==void 0&&(Ai("THREE.WebGLRenderTarget: option.encoding has been replaced by option.colorSpace."),n.colorSpace=n.encoding===gn?ot:Ct),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:yt,depthBuffer:!0,stencilBuffer:!1,depthTexture:null,samples:0},n),this.texture=new wt(i,n.mapping,n.wrapS,n.wrapT,n.magFilter,n.minFilter,n.format,n.type,n.anisotropy,n.colorSpace),this.texture.isRenderTargetTexture=!0,this.texture.flipY=!1,this.texture.generateMipmaps=n.generateMipmaps,this.texture.internalFormat=n.internalFormat,this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.depthTexture=n.depthTexture,this.samples=n.samples}setSize(t,e,n=1){(this.width!==t||this.height!==e||this.depth!==n)&&(this.width=t,this.height=e,this.depth=n,this.texture.image.width=t,this.texture.image.height=e,this.texture.image.depth=n,this.dispose()),this.viewport.set(0,0,t,e),this.scissor.set(0,0,t,e)}clone(){return new this.constructor().copy(this)}copy(t){this.width=t.width,this.height=t.height,this.depth=t.depth,this.scissor.copy(t.scissor),this.scissorTest=t.scissorTest,this.viewport.copy(t.viewport),this.texture=t.texture.clone(),this.texture.isRenderTargetTexture=!0;const e=Object.assign({},t.texture.image);return this.texture.source=new pa(e),this.depthBuffer=t.depthBuffer,this.stencilBuffer=t.stencilBuffer,t.depthTexture!==null&&(this.depthTexture=t.depthTexture.clone()),this.samples=t.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}},mn=class extends al{constructor(t=1,e=1,n={}){super(t,e,n),this.isWebGLRenderTarget=!0}},ga=class extends wt{constructor(t=null,e=1,n=1,i=1){super(null),this.isDataArrayTexture=!0,this.image={data:t,width:e,height:n,depth:i},this.magFilter=dt,this.minFilter=dt,this.wrapR=Bt,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}},ol=class extends wt{constructor(t=null,e=1,n=1,i=1){super(null),this.isData3DTexture=!0,this.image={data:t,width:e,height:n,depth:i},this.magFilter=dt,this.minFilter=dt,this.wrapR=Bt,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}},Dn=class{constructor(t=0,e=0,n=0,i=1){this.isQuaternion=!0,this._x=t,this._y=e,this._z=n,this._w=i}static slerpFlat(t,e,n,i,r,s,o){let a=n[i+0],A=n[i+1],l=n[i+2],c=n[i+3];const d=r[s+0],f=r[s+1],p=r[s+2],E=r[s+3];if(o===0){t[e+0]=a,t[e+1]=A,t[e+2]=l,t[e+3]=c;return}if(o===1){t[e+0]=d,t[e+1]=f,t[e+2]=p,t[e+3]=E;return}if(c!==E||a!==d||A!==f||l!==p){let g=1-o;const h=a*d+A*f+l*p+c*E,u=h>=0?1:-1,v=1-h*h;if(v>Number.EPSILON){const T=Math.sqrt(v),_=Math.atan2(T,h*u);g=Math.sin(g*_)/T,o=Math.sin(o*_)/T}const y=o*u;if(a=a*g+d*y,A=A*g+f*y,l=l*g+p*y,c=c*g+E*y,g===1-o){const T=1/Math.sqrt(a*a+A*A+l*l+c*c);a*=T,A*=T,l*=T,c*=T}}t[e]=a,t[e+1]=A,t[e+2]=l,t[e+3]=c}static multiplyQuaternionsFlat(t,e,n,i,r,s){const o=n[i],a=n[i+1],A=n[i+2],l=n[i+3],c=r[s],d=r[s+1],f=r[s+2],p=r[s+3];return t[e]=o*p+l*c+a*f-A*d,t[e+1]=a*p+l*d+A*c-o*f,t[e+2]=A*p+l*f+o*d-a*c,t[e+3]=l*p-o*c-a*d-A*f,t}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get w(){return this._w}set w(t){this._w=t,this._onChangeCallback()}set(t,e,n,i){return this._x=t,this._y=e,this._z=n,this._w=i,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(t){return this._x=t.x,this._y=t.y,this._z=t.z,this._w=t.w,this._onChangeCallback(),this}setFromEuler(t,e=!0){const n=t._x,i=t._y,r=t._z,s=t._order,o=Math.cos,a=Math.sin,A=o(n/2),l=o(i/2),c=o(r/2),d=a(n/2),f=a(i/2),p=a(r/2);switch(s){case"XYZ":this._x=d*l*c+A*f*p,this._y=A*f*c-d*l*p,this._z=A*l*p+d*f*c,this._w=A*l*c-d*f*p;break;case"YXZ":this._x=d*l*c+A*f*p,this._y=A*f*c-d*l*p,this._z=A*l*p-d*f*c,this._w=A*l*c+d*f*p;break;case"ZXY":this._x=d*l*c-A*f*p,this._y=A*f*c+d*l*p,this._z=A*l*p+d*f*c,this._w=A*l*c-d*f*p;break;case"ZYX":this._x=d*l*c-A*f*p,this._y=A*f*c+d*l*p,this._z=A*l*p-d*f*c,this._w=A*l*c+d*f*p;break;case"YZX":this._x=d*l*c+A*f*p,this._y=A*f*c+d*l*p,this._z=A*l*p-d*f*c,this._w=A*l*c-d*f*p;break;case"XZY":this._x=d*l*c-A*f*p,this._y=A*f*c-d*l*p,this._z=A*l*p+d*f*c,this._w=A*l*c+d*f*p;break;default:console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: "+s)}return e===!0&&this._onChangeCallback(),this}setFromAxisAngle(t,e){const n=e/2,i=Math.sin(n);return this._x=t.x*i,this._y=t.y*i,this._z=t.z*i,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(t){const e=t.elements,n=e[0],i=e[4],r=e[8],s=e[1],o=e[5],a=e[9],A=e[2],l=e[6],c=e[10],d=n+o+c;if(d>0){const f=.5/Math.sqrt(d+1);this._w=.25/f,this._x=(l-a)*f,this._y=(r-A)*f,this._z=(s-i)*f}else if(n>o&&n>c){const f=2*Math.sqrt(1+n-o-c);this._w=(l-a)/f,this._x=.25*f,this._y=(i+s)/f,this._z=(r+A)/f}else if(o>c){const f=2*Math.sqrt(1+o-n-c);this._w=(r-A)/f,this._x=(i+s)/f,this._y=.25*f,this._z=(a+l)/f}else{const f=2*Math.sqrt(1+c-n-o);this._w=(s-i)/f,this._x=(r+A)/f,this._y=(a+l)/f,this._z=.25*f}return this._onChangeCallback(),this}setFromUnitVectors(t,e){let n=t.dot(e)+1;return n<Number.EPSILON?(n=0,Math.abs(t.x)>Math.abs(t.z)?(this._x=-t.y,this._y=t.x,this._z=0,this._w=n):(this._x=0,this._y=-t.z,this._z=t.y,this._w=n)):(this._x=t.y*e.z-t.z*e.y,this._y=t.z*e.x-t.x*e.z,this._z=t.x*e.y-t.y*e.x,this._w=n),this.normalize()}angleTo(t){return 2*Math.acos(Math.abs(ft(this.dot(t),-1,1)))}rotateTowards(t,e){const n=this.angleTo(t);if(n===0)return this;const i=Math.min(1,e/n);return this.slerp(t,i),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(t){return this._x*t._x+this._y*t._y+this._z*t._z+this._w*t._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let t=this.length();return t===0?(this._x=0,this._y=0,this._z=0,this._w=1):(t=1/t,this._x=this._x*t,this._y=this._y*t,this._z=this._z*t,this._w=this._w*t),this._onChangeCallback(),this}multiply(t){return this.multiplyQuaternions(this,t)}premultiply(t){return this.multiplyQuaternions(t,this)}multiplyQuaternions(t,e){const n=t._x,i=t._y,r=t._z,s=t._w,o=e._x,a=e._y,A=e._z,l=e._w;return this._x=n*l+s*o+i*A-r*a,this._y=i*l+s*a+r*o-n*A,this._z=r*l+s*A+n*a-i*o,this._w=s*l-n*o-i*a-r*A,this._onChangeCallback(),this}slerp(t,e){if(e===0)return this;if(e===1)return this.copy(t);const n=this._x,i=this._y,r=this._z,s=this._w;let o=s*t._w+n*t._x+i*t._y+r*t._z;if(o<0?(this._w=-t._w,this._x=-t._x,this._y=-t._y,this._z=-t._z,o=-o):this.copy(t),o>=1)return this._w=s,this._x=n,this._y=i,this._z=r,this;const a=1-o*o;if(a<=Number.EPSILON){const f=1-e;return this._w=f*s+e*this._w,this._x=f*n+e*this._x,this._y=f*i+e*this._y,this._z=f*r+e*this._z,this.normalize(),this}const A=Math.sqrt(a),l=Math.atan2(A,o),c=Math.sin((1-e)*l)/A,d=Math.sin(e*l)/A;return this._w=s*c+this._w*d,this._x=n*c+this._x*d,this._y=i*c+this._y*d,this._z=r*c+this._z*d,this._onChangeCallback(),this}slerpQuaternions(t,e,n){return this.copy(t).slerp(e,n)}random(){const t=Math.random(),e=Math.sqrt(1-t),n=Math.sqrt(t),i=2*Math.PI*Math.random(),r=2*Math.PI*Math.random();return this.set(e*Math.cos(i),n*Math.sin(r),n*Math.cos(r),e*Math.sin(i))}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._w===this._w}fromArray(t,e=0){return this._x=t[e],this._y=t[e+1],this._z=t[e+2],this._w=t[e+3],this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._w,t}fromBufferAttribute(t,e){return this._x=t.getX(e),this._y=t.getY(e),this._z=t.getZ(e),this._w=t.getW(e),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},J=class ko{constructor(e=0,n=0,i=0){ko.prototype.isVector3=!0,this.x=e,this.y=n,this.z=i}set(e,n,i){return i===void 0&&(i=this.z),this.x=e,this.y=n,this.z=i,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,n){switch(e){case 0:this.x=n;break;case 1:this.y=n;break;case 2:this.z=n;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,n){return this.x=e.x+n.x,this.y=e.y+n.y,this.z=e.z+n.z,this}addScaledVector(e,n){return this.x+=e.x*n,this.y+=e.y*n,this.z+=e.z*n,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,n){return this.x=e.x-n.x,this.y=e.y-n.y,this.z=e.z-n.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,n){return this.x=e.x*n.x,this.y=e.y*n.y,this.z=e.z*n.z,this}applyEuler(e){return this.applyQuaternion(ma.setFromEuler(e))}applyAxisAngle(e,n){return this.applyQuaternion(ma.setFromAxisAngle(e,n))}applyMatrix3(e){const n=this.x,i=this.y,r=this.z,s=e.elements;return this.x=s[0]*n+s[3]*i+s[6]*r,this.y=s[1]*n+s[4]*i+s[7]*r,this.z=s[2]*n+s[5]*i+s[8]*r,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const n=this.x,i=this.y,r=this.z,s=e.elements,o=1/(s[3]*n+s[7]*i+s[11]*r+s[15]);return this.x=(s[0]*n+s[4]*i+s[8]*r+s[12])*o,this.y=(s[1]*n+s[5]*i+s[9]*r+s[13])*o,this.z=(s[2]*n+s[6]*i+s[10]*r+s[14])*o,this}applyQuaternion(e){const n=this.x,i=this.y,r=this.z,s=e.x,o=e.y,a=e.z,A=e.w,l=2*(o*r-a*i),c=2*(a*n-s*r),d=2*(s*i-o*n);return this.x=n+A*l+o*d-a*c,this.y=i+A*c+a*l-s*d,this.z=r+A*d+s*c-o*l,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const n=this.x,i=this.y,r=this.z,s=e.elements;return this.x=s[0]*n+s[4]*i+s[8]*r,this.y=s[1]*n+s[5]*i+s[9]*r,this.z=s[2]*n+s[6]*i+s[10]*r,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,n){return this.x=Math.max(e.x,Math.min(n.x,this.x)),this.y=Math.max(e.y,Math.min(n.y,this.y)),this.z=Math.max(e.z,Math.min(n.z,this.z)),this}clampScalar(e,n){return this.x=Math.max(e,Math.min(n,this.x)),this.y=Math.max(e,Math.min(n,this.y)),this.z=Math.max(e,Math.min(n,this.z)),this}clampLength(e,n){const i=this.length();return this.divideScalar(i||1).multiplyScalar(Math.max(e,Math.min(n,i)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,n){return this.x+=(e.x-this.x)*n,this.y+=(e.y-this.y)*n,this.z+=(e.z-this.z)*n,this}lerpVectors(e,n,i){return this.x=e.x+(n.x-e.x)*i,this.y=e.y+(n.y-e.y)*i,this.z=e.z+(n.z-e.z)*i,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,n){const i=e.x,r=e.y,s=e.z,o=n.x,a=n.y,A=n.z;return this.x=r*A-s*a,this.y=s*o-i*A,this.z=i*a-r*o,this}projectOnVector(e){const n=e.lengthSq();if(n===0)return this.set(0,0,0);const i=e.dot(this)/n;return this.copy(e).multiplyScalar(i)}projectOnPlane(e){return wr.copy(this).projectOnVector(e),this.sub(wr)}reflect(e){return this.sub(wr.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const n=Math.sqrt(this.lengthSq()*e.lengthSq());if(n===0)return Math.PI/2;const i=this.dot(e)/n;return Math.acos(ft(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const n=this.x-e.x,i=this.y-e.y,r=this.z-e.z;return n*n+i*i+r*r}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,n,i){const r=Math.sin(n)*e;return this.x=r*Math.sin(i),this.y=Math.cos(n)*e,this.z=r*Math.cos(i),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,n,i){return this.x=e*Math.sin(n),this.y=i,this.z=e*Math.cos(n),this}setFromMatrixPosition(e){const n=e.elements;return this.x=n[12],this.y=n[13],this.z=n[14],this}setFromMatrixScale(e){const n=this.setFromMatrixColumn(e,0).length(),i=this.setFromMatrixColumn(e,1).length(),r=this.setFromMatrixColumn(e,2).length();return this.x=n,this.y=i,this.z=r,this}setFromMatrixColumn(e,n){return this.fromArray(e.elements,n*4)}setFromMatrix3Column(e,n){return this.fromArray(e.elements,n*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,n=0){return this.x=e[n],this.y=e[n+1],this.z=e[n+2],this}toArray(e=[],n=0){return e[n]=this.x,e[n+1]=this.y,e[n+2]=this.z,e}fromBufferAttribute(e,n){return this.x=e.getX(n),this.y=e.getY(n),this.z=e.getZ(n),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=(Math.random()-.5)*2,n=Math.random()*Math.PI*2,i=Math.sqrt(1-e**2);return this.x=i*Math.cos(n),this.y=i*Math.sin(n),this.z=e,this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}},wr=new J,ma=new Dn,li=class{constructor(t=new J(1/0,1/0,1/0),e=new J(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=t,this.max=e}set(t,e){return this.min.copy(t),this.max.copy(e),this}setFromArray(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e+=3)this.expandByPoint(Rt.fromArray(t,e));return this}setFromBufferAttribute(t){this.makeEmpty();for(let e=0,n=t.count;e<n;e++)this.expandByPoint(Rt.fromBufferAttribute(t,e));return this}setFromPoints(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e++)this.expandByPoint(t[e]);return this}setFromCenterAndSize(t,e){const n=Rt.copy(e).multiplyScalar(.5);return this.min.copy(t).sub(n),this.max.copy(t).add(n),this}setFromObject(t,e=!1){return this.makeEmpty(),this.expandByObject(t,e)}clone(){return new this.constructor().copy(this)}copy(t){return this.min.copy(t.min),this.max.copy(t.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(t){return this.isEmpty()?t.set(0,0,0):t.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(t){return this.isEmpty()?t.set(0,0,0):t.subVectors(this.max,this.min)}expandByPoint(t){return this.min.min(t),this.max.max(t),this}expandByVector(t){return this.min.sub(t),this.max.add(t),this}expandByScalar(t){return this.min.addScalar(-t),this.max.addScalar(t),this}expandByObject(t,e=!1){t.updateWorldMatrix(!1,!1);const n=t.geometry;if(n!==void 0){const r=n.getAttribute("position");if(e===!0&&r!==void 0&&t.isInstancedMesh!==!0)for(let s=0,o=r.count;s<o;s++)t.isMesh===!0?t.getVertexPosition(s,Rt):Rt.fromBufferAttribute(r,s),Rt.applyMatrix4(t.matrixWorld),this.expandByPoint(Rt);else t.boundingBox!==void 0?(t.boundingBox===null&&t.computeBoundingBox(),Ri.copy(t.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),Ri.copy(n.boundingBox)),Ri.applyMatrix4(t.matrixWorld),this.union(Ri)}const i=t.children;for(let r=0,s=i.length;r<s;r++)this.expandByObject(i[r],e);return this}containsPoint(t){return!(t.x<this.min.x||t.x>this.max.x||t.y<this.min.y||t.y>this.max.y||t.z<this.min.z||t.z>this.max.z)}containsBox(t){return this.min.x<=t.min.x&&t.max.x<=this.max.x&&this.min.y<=t.min.y&&t.max.y<=this.max.y&&this.min.z<=t.min.z&&t.max.z<=this.max.z}getParameter(t,e){return e.set((t.x-this.min.x)/(this.max.x-this.min.x),(t.y-this.min.y)/(this.max.y-this.min.y),(t.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(t){return!(t.max.x<this.min.x||t.min.x>this.max.x||t.max.y<this.min.y||t.min.y>this.max.y||t.max.z<this.min.z||t.min.z>this.max.z)}intersectsSphere(t){return this.clampPoint(t.center,Rt),Rt.distanceToSquared(t.center)<=t.radius*t.radius}intersectsPlane(t){let e,n;return t.normal.x>0?(e=t.normal.x*this.min.x,n=t.normal.x*this.max.x):(e=t.normal.x*this.max.x,n=t.normal.x*this.min.x),t.normal.y>0?(e+=t.normal.y*this.min.y,n+=t.normal.y*this.max.y):(e+=t.normal.y*this.max.y,n+=t.normal.y*this.min.y),t.normal.z>0?(e+=t.normal.z*this.min.z,n+=t.normal.z*this.max.z):(e+=t.normal.z*this.max.z,n+=t.normal.z*this.min.z),e<=-t.constant&&n>=-t.constant}intersectsTriangle(t){if(this.isEmpty())return!1;this.getCenter(ci),_i.subVectors(this.max,ci),Hn.subVectors(t.a,ci),Qn.subVectors(t.b,ci),Nn.subVectors(t.c,ci),an.subVectors(Qn,Hn),on.subVectors(Nn,Qn),En.subVectors(Hn,Nn);let e=[0,-an.z,an.y,0,-on.z,on.y,0,-En.z,En.y,an.z,0,-an.x,on.z,0,-on.x,En.z,0,-En.x,-an.y,an.x,0,-on.y,on.x,0,-En.y,En.x,0];return!Jr(e,Hn,Qn,Nn,_i)||(e=[1,0,0,0,1,0,0,0,1],!Jr(e,Hn,Qn,Nn,_i))?!1:(Li.crossVectors(an,on),e=[Li.x,Li.y,Li.z],Jr(e,Hn,Qn,Nn,_i))}clampPoint(t,e){return e.copy(t).clamp(this.min,this.max)}distanceToPoint(t){return this.clampPoint(t,Rt).distanceTo(t)}getBoundingSphere(t){return this.isEmpty()?t.makeEmpty():(this.getCenter(t.center),t.radius=this.getSize(Rt).length()*.5),t}intersect(t){return this.min.max(t.min),this.max.min(t.max),this.isEmpty()&&this.makeEmpty(),this}union(t){return this.min.min(t.min),this.max.max(t.max),this}applyMatrix4(t){return this.isEmpty()?this:(Ut[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(t),Ut[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(t),Ut[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(t),Ut[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(t),Ut[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(t),Ut[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(t),Ut[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(t),Ut[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(t),this.setFromPoints(Ut),this)}translate(t){return this.min.add(t),this.max.add(t),this}equals(t){return t.min.equals(this.min)&&t.max.equals(this.max)}},Ut=[new J,new J,new J,new J,new J,new J,new J,new J],Rt=new J,Ri=new li,Hn=new J,Qn=new J,Nn=new J,an=new J,on=new J,En=new J,ci=new J,_i=new J,Li=new J,Sn=new J;function Jr(t,e,n,i,r){for(let s=0,o=t.length-3;s<=o;s+=3){Sn.fromArray(t,s);const a=r.x*Math.abs(Sn.x)+r.y*Math.abs(Sn.y)+r.z*Math.abs(Sn.z),A=e.dot(Sn),l=n.dot(Sn),c=i.dot(Sn);if(Math.max(-Math.max(A,l,c),Math.min(A,l,c))>a)return!1}return!0}var Al=new li,ui=new J,Or=new J,wi=class{constructor(t=new J,e=-1){this.isSphere=!0,this.center=t,this.radius=e}set(t,e){return this.center.copy(t),this.radius=e,this}setFromPoints(t,e){const n=this.center;e!==void 0?n.copy(e):Al.setFromPoints(t).getCenter(n);let i=0;for(let r=0,s=t.length;r<s;r++)i=Math.max(i,n.distanceToSquared(t[r]));return this.radius=Math.sqrt(i),this}copy(t){return this.center.copy(t.center),this.radius=t.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(t){return t.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(t){return t.distanceTo(this.center)-this.radius}intersectsSphere(t){const e=this.radius+t.radius;return t.center.distanceToSquared(this.center)<=e*e}intersectsBox(t){return t.intersectsSphere(this)}intersectsPlane(t){return Math.abs(t.distanceToPoint(this.center))<=this.radius}clampPoint(t,e){const n=this.center.distanceToSquared(t);return e.copy(t),n>this.radius*this.radius&&(e.sub(this.center).normalize(),e.multiplyScalar(this.radius).add(this.center)),e}getBoundingBox(t){return this.isEmpty()?(t.makeEmpty(),t):(t.set(this.center,this.center),t.expandByScalar(this.radius),t)}applyMatrix4(t){return this.center.applyMatrix4(t),this.radius=this.radius*t.getMaxScaleOnAxis(),this}translate(t){return this.center.add(t),this}expandByPoint(t){if(this.isEmpty())return this.center.copy(t),this.radius=0,this;ui.subVectors(t,this.center);const e=ui.lengthSq();if(e>this.radius*this.radius){const n=Math.sqrt(e),i=(n-this.radius)*.5;this.center.addScaledVector(ui,i/n),this.radius+=i}return this}union(t){return t.isEmpty()?this:this.isEmpty()?(this.copy(t),this):(this.center.equals(t.center)===!0?this.radius=Math.max(this.radius,t.radius):(Or.subVectors(t.center,this.center).setLength(t.radius),this.expandByPoint(ui.copy(t.center).add(Or)),this.expandByPoint(ui.copy(t.center).sub(Or))),this)}equals(t){return t.center.equals(this.center)&&t.radius===this.radius}clone(){return new this.constructor().copy(this)}},Xt=new J,Dr=new J,Ji=new J,An=new J,Hr=new J,Oi=new J,Qr=new J,Ea=class{constructor(t=new J,e=new J(0,0,-1)){this.origin=t,this.direction=e}set(t,e){return this.origin.copy(t),this.direction.copy(e),this}copy(t){return this.origin.copy(t.origin),this.direction.copy(t.direction),this}at(t,e){return e.copy(this.origin).addScaledVector(this.direction,t)}lookAt(t){return this.direction.copy(t).sub(this.origin).normalize(),this}recast(t){return this.origin.copy(this.at(t,Xt)),this}closestPointToPoint(t,e){e.subVectors(t,this.origin);const n=e.dot(this.direction);return n<0?e.copy(this.origin):e.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(t){return Math.sqrt(this.distanceSqToPoint(t))}distanceSqToPoint(t){const e=Xt.subVectors(t,this.origin).dot(this.direction);return e<0?this.origin.distanceToSquared(t):(Xt.copy(this.origin).addScaledVector(this.direction,e),Xt.distanceToSquared(t))}distanceSqToSegment(t,e,n,i){Dr.copy(t).add(e).multiplyScalar(.5),Ji.copy(e).sub(t).normalize(),An.copy(this.origin).sub(Dr);const r=t.distanceTo(e)*.5,s=-this.direction.dot(Ji),o=An.dot(this.direction),a=-An.dot(Ji),A=An.lengthSq(),l=Math.abs(1-s*s);let c,d,f,p;if(l>0)if(c=s*a-o,d=s*o-a,p=r*l,c>=0)if(d>=-p)if(d<=p){const E=1/l;c*=E,d*=E,f=c*(c+s*d+2*o)+d*(s*c+d+2*a)+A}else d=r,c=Math.max(0,-(s*d+o)),f=-c*c+d*(d+2*a)+A;else d=-r,c=Math.max(0,-(s*d+o)),f=-c*c+d*(d+2*a)+A;else d<=-p?(c=Math.max(0,-(-s*r+o)),d=c>0?-r:Math.min(Math.max(-r,-a),r),f=-c*c+d*(d+2*a)+A):d<=p?(c=0,d=Math.min(Math.max(-r,-a),r),f=d*(d+2*a)+A):(c=Math.max(0,-(s*r+o)),d=c>0?r:Math.min(Math.max(-r,-a),r),f=-c*c+d*(d+2*a)+A);else d=s>0?-r:r,c=Math.max(0,-(s*d+o)),f=-c*c+d*(d+2*a)+A;return n&&n.copy(this.origin).addScaledVector(this.direction,c),i&&i.copy(Dr).addScaledVector(Ji,d),f}intersectSphere(t,e){Xt.subVectors(t.center,this.origin);const n=Xt.dot(this.direction),i=Xt.dot(Xt)-n*n,r=t.radius*t.radius;if(i>r)return null;const s=Math.sqrt(r-i),o=n-s,a=n+s;return a<0?null:o<0?this.at(a,e):this.at(o,e)}intersectsSphere(t){return this.distanceSqToPoint(t.center)<=t.radius*t.radius}distanceToPlane(t){const e=t.normal.dot(this.direction);if(e===0)return t.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(t.normal)+t.constant)/e;return n>=0?n:null}intersectPlane(t,e){const n=this.distanceToPlane(t);return n===null?null:this.at(n,e)}intersectsPlane(t){const e=t.distanceToPoint(this.origin);return e===0||t.normal.dot(this.direction)*e<0}intersectBox(t,e){let n,i,r,s,o,a;const A=1/this.direction.x,l=1/this.direction.y,c=1/this.direction.z,d=this.origin;return A>=0?(n=(t.min.x-d.x)*A,i=(t.max.x-d.x)*A):(n=(t.max.x-d.x)*A,i=(t.min.x-d.x)*A),l>=0?(r=(t.min.y-d.y)*l,s=(t.max.y-d.y)*l):(r=(t.max.y-d.y)*l,s=(t.min.y-d.y)*l),n>s||r>i||((r>n||isNaN(n))&&(n=r),(s<i||isNaN(i))&&(i=s),c>=0?(o=(t.min.z-d.z)*c,a=(t.max.z-d.z)*c):(o=(t.max.z-d.z)*c,a=(t.min.z-d.z)*c),n>a||o>i)||((o>n||n!==n)&&(n=o),(a<i||i!==i)&&(i=a),i<0)?null:this.at(n>=0?n:i,e)}intersectsBox(t){return this.intersectBox(t,Xt)!==null}intersectTriangle(t,e,n,i,r){Hr.subVectors(e,t),Oi.subVectors(n,t),Qr.crossVectors(Hr,Oi);let s=this.direction.dot(Qr),o;if(s>0){if(i)return null;o=1}else if(s<0)o=-1,s=-s;else return null;An.subVectors(this.origin,t);const a=o*this.direction.dot(Oi.crossVectors(An,Oi));if(a<0)return null;const A=o*this.direction.dot(Hr.cross(An));if(A<0||a+A>s)return null;const l=-o*An.dot(Qr);return l<0?null:this.at(l/s,r)}applyMatrix4(t){return this.origin.applyMatrix4(t),this.direction.transformDirection(t),this}equals(t){return t.origin.equals(this.origin)&&t.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},gt=class us{constructor(e,n,i,r,s,o,a,A,l,c,d,f,p,E,g,h){us.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,n,i,r,s,o,a,A,l,c,d,f,p,E,g,h)}set(e,n,i,r,s,o,a,A,l,c,d,f,p,E,g,h){const u=this.elements;return u[0]=e,u[4]=n,u[8]=i,u[12]=r,u[1]=s,u[5]=o,u[9]=a,u[13]=A,u[2]=l,u[6]=c,u[10]=d,u[14]=f,u[3]=p,u[7]=E,u[11]=g,u[15]=h,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new us().fromArray(this.elements)}copy(e){const n=this.elements,i=e.elements;return n[0]=i[0],n[1]=i[1],n[2]=i[2],n[3]=i[3],n[4]=i[4],n[5]=i[5],n[6]=i[6],n[7]=i[7],n[8]=i[8],n[9]=i[9],n[10]=i[10],n[11]=i[11],n[12]=i[12],n[13]=i[13],n[14]=i[14],n[15]=i[15],this}copyPosition(e){const n=this.elements,i=e.elements;return n[12]=i[12],n[13]=i[13],n[14]=i[14],this}setFromMatrix3(e){const n=e.elements;return this.set(n[0],n[3],n[6],0,n[1],n[4],n[7],0,n[2],n[5],n[8],0,0,0,0,1),this}extractBasis(e,n,i){return e.setFromMatrixColumn(this,0),n.setFromMatrixColumn(this,1),i.setFromMatrixColumn(this,2),this}makeBasis(e,n,i){return this.set(e.x,n.x,i.x,0,e.y,n.y,i.y,0,e.z,n.z,i.z,0,0,0,0,1),this}extractRotation(e){const n=this.elements,i=e.elements,r=1/Gn.setFromMatrixColumn(e,0).length(),s=1/Gn.setFromMatrixColumn(e,1).length(),o=1/Gn.setFromMatrixColumn(e,2).length();return n[0]=i[0]*r,n[1]=i[1]*r,n[2]=i[2]*r,n[3]=0,n[4]=i[4]*s,n[5]=i[5]*s,n[6]=i[6]*s,n[7]=0,n[8]=i[8]*o,n[9]=i[9]*o,n[10]=i[10]*o,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromEuler(e){const n=this.elements,i=e.x,r=e.y,s=e.z,o=Math.cos(i),a=Math.sin(i),A=Math.cos(r),l=Math.sin(r),c=Math.cos(s),d=Math.sin(s);if(e.order==="XYZ"){const f=o*c,p=o*d,E=a*c,g=a*d;n[0]=A*c,n[4]=-A*d,n[8]=l,n[1]=p+E*l,n[5]=f-g*l,n[9]=-a*A,n[2]=g-f*l,n[6]=E+p*l,n[10]=o*A}else if(e.order==="YXZ"){const f=A*c,p=A*d,E=l*c,g=l*d;n[0]=f+g*a,n[4]=E*a-p,n[8]=o*l,n[1]=o*d,n[5]=o*c,n[9]=-a,n[2]=p*a-E,n[6]=g+f*a,n[10]=o*A}else if(e.order==="ZXY"){const f=A*c,p=A*d,E=l*c,g=l*d;n[0]=f-g*a,n[4]=-o*d,n[8]=E+p*a,n[1]=p+E*a,n[5]=o*c,n[9]=g-f*a,n[2]=-o*l,n[6]=a,n[10]=o*A}else if(e.order==="ZYX"){const f=o*c,p=o*d,E=a*c,g=a*d;n[0]=A*c,n[4]=E*l-p,n[8]=f*l+g,n[1]=A*d,n[5]=g*l+f,n[9]=p*l-E,n[2]=-l,n[6]=a*A,n[10]=o*A}else if(e.order==="YZX"){const f=o*A,p=o*l,E=a*A,g=a*l;n[0]=A*c,n[4]=g-f*d,n[8]=E*d+p,n[1]=d,n[5]=o*c,n[9]=-a*c,n[2]=-l*c,n[6]=p*d+E,n[10]=f-g*d}else if(e.order==="XZY"){const f=o*A,p=o*l,E=a*A,g=a*l;n[0]=A*c,n[4]=-d,n[8]=l*c,n[1]=f*d+g,n[5]=o*c,n[9]=p*d-E,n[2]=E*d-p,n[6]=a*c,n[10]=g*d+f}return n[3]=0,n[7]=0,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,this}makeRotationFromQuaternion(e){return this.compose(ll,e,cl)}lookAt(e,n,i){const r=this.elements;return xt.subVectors(e,n),xt.lengthSq()===0&&(xt.z=1),xt.normalize(),ln.crossVectors(i,xt),ln.lengthSq()===0&&(Math.abs(i.z)===1?xt.x+=1e-4:xt.z+=1e-4,xt.normalize(),ln.crossVectors(i,xt)),ln.normalize(),Di.crossVectors(xt,ln),r[0]=ln.x,r[4]=Di.x,r[8]=xt.x,r[1]=ln.y,r[5]=Di.y,r[9]=xt.y,r[2]=ln.z,r[6]=Di.z,r[10]=xt.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,n){const i=e.elements,r=n.elements,s=this.elements,o=i[0],a=i[4],A=i[8],l=i[12],c=i[1],d=i[5],f=i[9],p=i[13],E=i[2],g=i[6],h=i[10],u=i[14],v=i[3],y=i[7],T=i[11],_=i[15],C=r[0],B=r[4],z=r[8],M=r[12],I=r[1],H=r[5],W=r[9],Y=r[13],b=r[2],Q=r[6],G=r[10],q=r[14],U=r[3],j=r[7],X=r[11],ee=r[15];return s[0]=o*C+a*I+A*b+l*U,s[4]=o*B+a*H+A*Q+l*j,s[8]=o*z+a*W+A*G+l*X,s[12]=o*M+a*Y+A*q+l*ee,s[1]=c*C+d*I+f*b+p*U,s[5]=c*B+d*H+f*Q+p*j,s[9]=c*z+d*W+f*G+p*X,s[13]=c*M+d*Y+f*q+p*ee,s[2]=E*C+g*I+h*b+u*U,s[6]=E*B+g*H+h*Q+u*j,s[10]=E*z+g*W+h*G+u*X,s[14]=E*M+g*Y+h*q+u*ee,s[3]=v*C+y*I+T*b+_*U,s[7]=v*B+y*H+T*Q+_*j,s[11]=v*z+y*W+T*G+_*X,s[15]=v*M+y*Y+T*q+_*ee,this}multiplyScalar(e){const n=this.elements;return n[0]*=e,n[4]*=e,n[8]*=e,n[12]*=e,n[1]*=e,n[5]*=e,n[9]*=e,n[13]*=e,n[2]*=e,n[6]*=e,n[10]*=e,n[14]*=e,n[3]*=e,n[7]*=e,n[11]*=e,n[15]*=e,this}determinant(){const e=this.elements,n=e[0],i=e[4],r=e[8],s=e[12],o=e[1],a=e[5],A=e[9],l=e[13],c=e[2],d=e[6],f=e[10],p=e[14],E=e[3],g=e[7],h=e[11],u=e[15];return E*(+s*A*d-r*l*d-s*a*f+i*l*f+r*a*p-i*A*p)+g*(+n*A*p-n*l*f+s*o*f-r*o*p+r*l*c-s*A*c)+h*(+n*l*d-n*a*p-s*o*d+i*o*p+s*a*c-i*l*c)+u*(-r*a*c-n*A*d+n*a*f+r*o*d-i*o*f+i*A*c)}transpose(){const e=this.elements;let n;return n=e[1],e[1]=e[4],e[4]=n,n=e[2],e[2]=e[8],e[8]=n,n=e[6],e[6]=e[9],e[9]=n,n=e[3],e[3]=e[12],e[12]=n,n=e[7],e[7]=e[13],e[13]=n,n=e[11],e[11]=e[14],e[14]=n,this}setPosition(e,n,i){const r=this.elements;return e.isVector3?(r[12]=e.x,r[13]=e.y,r[14]=e.z):(r[12]=e,r[13]=n,r[14]=i),this}invert(){const e=this.elements,n=e[0],i=e[1],r=e[2],s=e[3],o=e[4],a=e[5],A=e[6],l=e[7],c=e[8],d=e[9],f=e[10],p=e[11],E=e[12],g=e[13],h=e[14],u=e[15],v=d*h*l-g*f*l+g*A*p-a*h*p-d*A*u+a*f*u,y=E*f*l-c*h*l-E*A*p+o*h*p+c*A*u-o*f*u,T=c*g*l-E*d*l+E*a*p-o*g*p-c*a*u+o*d*u,_=E*d*A-c*g*A-E*a*f+o*g*f+c*a*h-o*d*h,C=n*v+i*y+r*T+s*_;if(C===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const B=1/C;return e[0]=v*B,e[1]=(g*f*s-d*h*s-g*r*p+i*h*p+d*r*u-i*f*u)*B,e[2]=(a*h*s-g*A*s+g*r*l-i*h*l-a*r*u+i*A*u)*B,e[3]=(d*A*s-a*f*s-d*r*l+i*f*l+a*r*p-i*A*p)*B,e[4]=y*B,e[5]=(c*h*s-E*f*s+E*r*p-n*h*p-c*r*u+n*f*u)*B,e[6]=(E*A*s-o*h*s-E*r*l+n*h*l+o*r*u-n*A*u)*B,e[7]=(o*f*s-c*A*s+c*r*l-n*f*l-o*r*p+n*A*p)*B,e[8]=T*B,e[9]=(E*d*s-c*g*s-E*i*p+n*g*p+c*i*u-n*d*u)*B,e[10]=(o*g*s-E*a*s+E*i*l-n*g*l-o*i*u+n*a*u)*B,e[11]=(c*a*s-o*d*s-c*i*l+n*d*l+o*i*p-n*a*p)*B,e[12]=_*B,e[13]=(c*g*r-E*d*r+E*i*f-n*g*f-c*i*h+n*d*h)*B,e[14]=(E*a*r-o*g*r-E*i*A+n*g*A+o*i*h-n*a*h)*B,e[15]=(o*d*r-c*a*r+c*i*A-n*d*A-o*i*f+n*a*f)*B,this}scale(e){const n=this.elements,i=e.x,r=e.y,s=e.z;return n[0]*=i,n[4]*=r,n[8]*=s,n[1]*=i,n[5]*=r,n[9]*=s,n[2]*=i,n[6]*=r,n[10]*=s,n[3]*=i,n[7]*=r,n[11]*=s,this}getMaxScaleOnAxis(){const e=this.elements,n=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],i=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],r=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(n,i,r))}makeTranslation(e,n,i){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,n,0,0,1,i,0,0,0,1),this}makeRotationX(e){const n=Math.cos(e),i=Math.sin(e);return this.set(1,0,0,0,0,n,-i,0,0,i,n,0,0,0,0,1),this}makeRotationY(e){const n=Math.cos(e),i=Math.sin(e);return this.set(n,0,i,0,0,1,0,0,-i,0,n,0,0,0,0,1),this}makeRotationZ(e){const n=Math.cos(e),i=Math.sin(e);return this.set(n,-i,0,0,i,n,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,n){const i=Math.cos(n),r=Math.sin(n),s=1-i,o=e.x,a=e.y,A=e.z,l=s*o,c=s*a;return this.set(l*o+i,l*a-r*A,l*A+r*a,0,l*a+r*A,c*a+i,c*A-r*o,0,l*A-r*a,c*A+r*o,s*A*A+i,0,0,0,0,1),this}makeScale(e,n,i){return this.set(e,0,0,0,0,n,0,0,0,0,i,0,0,0,0,1),this}makeShear(e,n,i,r,s,o){return this.set(1,i,s,0,e,1,o,0,n,r,1,0,0,0,0,1),this}compose(e,n,i){const r=this.elements,s=n._x,o=n._y,a=n._z,A=n._w,l=s+s,c=o+o,d=a+a,f=s*l,p=s*c,E=s*d,g=o*c,h=o*d,u=a*d,v=A*l,y=A*c,T=A*d,_=i.x,C=i.y,B=i.z;return r[0]=(1-(g+u))*_,r[1]=(p+T)*_,r[2]=(E-y)*_,r[3]=0,r[4]=(p-T)*C,r[5]=(1-(f+u))*C,r[6]=(h+v)*C,r[7]=0,r[8]=(E+y)*B,r[9]=(h-v)*B,r[10]=(1-(f+g))*B,r[11]=0,r[12]=e.x,r[13]=e.y,r[14]=e.z,r[15]=1,this}decompose(e,n,i){const r=this.elements;let s=Gn.set(r[0],r[1],r[2]).length();const o=Gn.set(r[4],r[5],r[6]).length(),a=Gn.set(r[8],r[9],r[10]).length();this.determinant()<0&&(s=-s),e.x=r[12],e.y=r[13],e.z=r[14],_t.copy(this);const l=1/s,c=1/o,d=1/a;return _t.elements[0]*=l,_t.elements[1]*=l,_t.elements[2]*=l,_t.elements[4]*=c,_t.elements[5]*=c,_t.elements[6]*=c,_t.elements[8]*=d,_t.elements[9]*=d,_t.elements[10]*=d,n.setFromRotationMatrix(_t),i.x=s,i.y=o,i.z=a,this}makePerspective(e,n,i,r,s,o,a=jt){const A=this.elements,l=2*s/(n-e),c=2*s/(i-r),d=(n+e)/(n-e),f=(i+r)/(i-r);let p,E;if(a===jt)p=-(o+s)/(o-s),E=-2*o*s/(o-s);else if(a===Pi)p=-o/(o-s),E=-o*s/(o-s);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return A[0]=l,A[4]=0,A[8]=d,A[12]=0,A[1]=0,A[5]=c,A[9]=f,A[13]=0,A[2]=0,A[6]=0,A[10]=p,A[14]=E,A[3]=0,A[7]=0,A[11]=-1,A[15]=0,this}makeOrthographic(e,n,i,r,s,o,a=jt){const A=this.elements,l=1/(n-e),c=1/(i-r),d=1/(o-s),f=(n+e)*l,p=(i+r)*c;let E,g;if(a===jt)E=(o+s)*d,g=-2*d;else if(a===Pi)E=s*d,g=-1*d;else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return A[0]=2*l,A[4]=0,A[8]=0,A[12]=-f,A[1]=0,A[5]=2*c,A[9]=0,A[13]=-p,A[2]=0,A[6]=0,A[10]=g,A[14]=-E,A[3]=0,A[7]=0,A[11]=0,A[15]=1,this}equals(e){const n=this.elements,i=e.elements;for(let r=0;r<16;r++)if(n[r]!==i[r])return!1;return!0}fromArray(e,n=0){for(let i=0;i<16;i++)this.elements[i]=e[i+n];return this}toArray(e=[],n=0){const i=this.elements;return e[n]=i[0],e[n+1]=i[1],e[n+2]=i[2],e[n+3]=i[3],e[n+4]=i[4],e[n+5]=i[5],e[n+6]=i[6],e[n+7]=i[7],e[n+8]=i[8],e[n+9]=i[9],e[n+10]=i[10],e[n+11]=i[11],e[n+12]=i[12],e[n+13]=i[13],e[n+14]=i[14],e[n+15]=i[15],e}},Gn=new J,_t=new gt,ll=new J(0,0,0),cl=new J(1,1,1),ln=new J,Di=new J,xt=new J,Sa=new gt,va=new Dn,xa=class Ro{constructor(e=0,n=0,i=0,r=Ro.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=n,this._z=i,this._order=r}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,n,i,r=this._order){return this._x=e,this._y=n,this._z=i,this._order=r,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,n=this._order,i=!0){const r=e.elements,s=r[0],o=r[4],a=r[8],A=r[1],l=r[5],c=r[9],d=r[2],f=r[6],p=r[10];switch(n){case"XYZ":this._y=Math.asin(ft(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-c,p),this._z=Math.atan2(-o,s)):(this._x=Math.atan2(f,l),this._z=0);break;case"YXZ":this._x=Math.asin(-ft(c,-1,1)),Math.abs(c)<.9999999?(this._y=Math.atan2(a,p),this._z=Math.atan2(A,l)):(this._y=Math.atan2(-d,s),this._z=0);break;case"ZXY":this._x=Math.asin(ft(f,-1,1)),Math.abs(f)<.9999999?(this._y=Math.atan2(-d,p),this._z=Math.atan2(-o,l)):(this._y=0,this._z=Math.atan2(A,s));break;case"ZYX":this._y=Math.asin(-ft(d,-1,1)),Math.abs(d)<.9999999?(this._x=Math.atan2(f,p),this._z=Math.atan2(A,s)):(this._x=0,this._z=Math.atan2(-o,l));break;case"YZX":this._z=Math.asin(ft(A,-1,1)),Math.abs(A)<.9999999?(this._x=Math.atan2(-c,l),this._y=Math.atan2(-d,s)):(this._x=0,this._y=Math.atan2(a,p));break;case"XZY":this._z=Math.asin(-ft(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(f,l),this._y=Math.atan2(a,s)):(this._x=Math.atan2(-c,p),this._y=0);break;default:console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: "+n)}return this._order=n,i===!0&&this._onChangeCallback(),this}setFromQuaternion(e,n,i){return Sa.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Sa,n,i)}setFromVector3(e,n=this._order){return this.set(e.x,e.y,e.z,n)}reorder(e){return va.setFromEuler(this),this.setFromQuaternion(va,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],n=0){return e[n]=this._x,e[n+1]=this._y,e[n+2]=this._z,e[n+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};xa.DEFAULT_ORDER="XYZ";var Ma=class{constructor(){this.mask=1}set(t){this.mask=(1<<t|0)>>>0}enable(t){this.mask|=1<<t|0}enableAll(){this.mask=-1}toggle(t){this.mask^=1<<t|0}disable(t){this.mask&=~(1<<t|0)}disableAll(){this.mask=0}test(t){return(this.mask&t.mask)!==0}isEnabled(t){return(this.mask&(1<<t|0))!==0}},ul=0,Ia=new J,zn=new Dn,Zt=new gt,Hi=new J,di=new J,dl=new J,fl=new Dn,ya=new J(1,0,0),Ca=new J(0,1,0),Ta=new J(0,0,1),hl={type:"added"},pl={type:"removed"},Jt=class lr extends _n{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:ul++}),this.uuid=Ln(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=lr.DEFAULT_UP.clone();const e=new J,n=new xa,i=new Dn,r=new J(1,1,1);function s(){i.setFromEuler(n,!1)}function o(){n.setFromQuaternion(i,void 0,!1)}n._onChange(s),i._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:n},quaternion:{configurable:!0,enumerable:!0,value:i},scale:{configurable:!0,enumerable:!0,value:r},modelViewMatrix:{value:new gt},normalMatrix:{value:new Oe}}),this.matrix=new gt,this.matrixWorld=new gt,this.matrixAutoUpdate=lr.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=lr.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Ma,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,n){this.quaternion.setFromAxisAngle(e,n)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,n){return zn.setFromAxisAngle(e,n),this.quaternion.multiply(zn),this}rotateOnWorldAxis(e,n){return zn.setFromAxisAngle(e,n),this.quaternion.premultiply(zn),this}rotateX(e){return this.rotateOnAxis(ya,e)}rotateY(e){return this.rotateOnAxis(Ca,e)}rotateZ(e){return this.rotateOnAxis(Ta,e)}translateOnAxis(e,n){return Ia.copy(e).applyQuaternion(this.quaternion),this.position.add(Ia.multiplyScalar(n)),this}translateX(e){return this.translateOnAxis(ya,e)}translateY(e){return this.translateOnAxis(Ca,e)}translateZ(e){return this.translateOnAxis(Ta,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Zt.copy(this.matrixWorld).invert())}lookAt(e,n,i){e.isVector3?Hi.copy(e):Hi.set(e,n,i);const r=this.parent;this.updateWorldMatrix(!0,!1),di.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Zt.lookAt(di,Hi,this.up):Zt.lookAt(Hi,di,this.up),this.quaternion.setFromRotationMatrix(Zt),r&&(Zt.extractRotation(r.matrixWorld),zn.setFromRotationMatrix(Zt),this.quaternion.premultiply(zn.invert()))}add(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.add(arguments[n]);return this}return e===this?(console.error("THREE.Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.parent!==null&&e.parent.remove(e),e.parent=this,this.children.push(e),e.dispatchEvent(hl)):console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}const n=this.children.indexOf(e);return n!==-1&&(e.parent=null,this.children.splice(n,1),e.dispatchEvent(pl)),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Zt.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Zt.multiply(e.parent.matrixWorld)),e.applyMatrix4(Zt),this.add(e),e.updateWorldMatrix(!1,!0),this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,n){if(this[e]===n)return this;for(let i=0,r=this.children.length;i<r;i++){const o=this.children[i].getObjectByProperty(e,n);if(o!==void 0)return o}}getObjectsByProperty(e,n,i=[]){this[e]===n&&i.push(this);const r=this.children;for(let s=0,o=r.length;s<o;s++)r[s].getObjectsByProperty(e,n,i);return i}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(di,e,dl),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(di,fl,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const n=this.matrixWorld.elements;return e.set(n[8],n[9],n[10]).normalize()}raycast(){}traverse(e){e(this);const n=this.children;for(let i=0,r=n.length;i<r;i++)n[i].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const n=this.children;for(let i=0,r=n.length;i<r;i++)n[i].traverseVisible(e)}traverseAncestors(e){const n=this.parent;n!==null&&(e(n),n.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix),this.matrixWorldNeedsUpdate=!1,e=!0);const n=this.children;for(let i=0,r=n.length;i<r;i++){const s=n[i];(s.matrixWorldAutoUpdate===!0||e===!0)&&s.updateMatrixWorld(e)}}updateWorldMatrix(e,n){const i=this.parent;if(e===!0&&i!==null&&i.matrixWorldAutoUpdate===!0&&i.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix),n===!0){const r=this.children;for(let s=0,o=r.length;s<o;s++){const a=r[s];a.matrixWorldAutoUpdate===!0&&a.updateWorldMatrix(!1,!0)}}}toJSON(e){const n=e===void 0||typeof e=="string",i={};n&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},i.metadata={version:4.6,type:"Object",generator:"Object3D.toJSON"});const r={};r.uuid=this.uuid,r.type=this.type,this.name!==""&&(r.name=this.name),this.castShadow===!0&&(r.castShadow=!0),this.receiveShadow===!0&&(r.receiveShadow=!0),this.visible===!1&&(r.visible=!1),this.frustumCulled===!1&&(r.frustumCulled=!1),this.renderOrder!==0&&(r.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(r.userData=this.userData),r.layers=this.layers.mask,r.matrix=this.matrix.toArray(),r.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(r.matrixAutoUpdate=!1),this.isInstancedMesh&&(r.type="InstancedMesh",r.count=this.count,r.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(r.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(r.type="BatchedMesh",r.perObjectFrustumCulled=this.perObjectFrustumCulled,r.sortObjects=this.sortObjects,r.drawRanges=this._drawRanges,r.reservedRanges=this._reservedRanges,r.visibility=this._visibility,r.active=this._active,r.bounds=this._bounds.map(a=>({boxInitialized:a.boxInitialized,boxMin:a.box.min.toArray(),boxMax:a.box.max.toArray(),sphereInitialized:a.sphereInitialized,sphereRadius:a.sphere.radius,sphereCenter:a.sphere.center.toArray()})),r.maxGeometryCount=this._maxGeometryCount,r.maxVertexCount=this._maxVertexCount,r.maxIndexCount=this._maxIndexCount,r.geometryInitialized=this._geometryInitialized,r.geometryCount=this._geometryCount,r.matricesTexture=this._matricesTexture.toJSON(e),this.boundingSphere!==null&&(r.boundingSphere={center:r.boundingSphere.center.toArray(),radius:r.boundingSphere.radius}),this.boundingBox!==null&&(r.boundingBox={min:r.boundingBox.min.toArray(),max:r.boundingBox.max.toArray()}));function s(a,A){return a[A.uuid]===void 0&&(a[A.uuid]=A.toJSON(e)),A.uuid}if(this.isScene)this.background&&(this.background.isColor?r.background=this.background.toJSON():this.background.isTexture&&(r.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(r.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){r.geometry=s(e.geometries,this.geometry);const a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){const A=a.shapes;if(Array.isArray(A))for(let l=0,c=A.length;l<c;l++){const d=A[l];s(e.shapes,d)}else s(e.shapes,A)}}if(this.isSkinnedMesh&&(r.bindMode=this.bindMode,r.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(s(e.skeletons,this.skeleton),r.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const a=[];for(let A=0,l=this.material.length;A<l;A++)a.push(s(e.materials,this.material[A]));r.material=a}else r.material=s(e.materials,this.material);if(this.children.length>0){r.children=[];for(let a=0;a<this.children.length;a++)r.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){r.animations=[];for(let a=0;a<this.animations.length;a++){const A=this.animations[a];r.animations.push(s(e.animations,A))}}if(n){const a=o(e.geometries),A=o(e.materials),l=o(e.textures),c=o(e.images),d=o(e.shapes),f=o(e.skeletons),p=o(e.animations),E=o(e.nodes);a.length>0&&(i.geometries=a),A.length>0&&(i.materials=A),l.length>0&&(i.textures=l),c.length>0&&(i.images=c),d.length>0&&(i.shapes=d),f.length>0&&(i.skeletons=f),p.length>0&&(i.animations=p),E.length>0&&(i.nodes=E)}return i.object=r,i;function o(a){const A=[];for(const l in a){const c=a[l];delete c.metadata,A.push(c)}return A}}clone(e){return new this.constructor().copy(this,e)}copy(e,n=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),n===!0)for(let i=0;i<e.children.length;i++){const r=e.children[i];this.add(r.clone())}return this}};Jt.DEFAULT_UP=new J(0,1,0),Jt.DEFAULT_MATRIX_AUTO_UPDATE=!0,Jt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var Lt=new J,Ft=new J,Nr=new J,Wt=new J,jn=new J,Un=new J,Pa=new J,Gr=new J,zr=new J,jr=new J,Qi=!1,Ni=class Pn{constructor(e=new J,n=new J,i=new J){this.a=e,this.b=n,this.c=i}static getNormal(e,n,i,r){r.subVectors(i,n),Lt.subVectors(e,n),r.cross(Lt);const s=r.lengthSq();return s>0?r.multiplyScalar(1/Math.sqrt(s)):r.set(0,0,0)}static getBarycoord(e,n,i,r,s){Lt.subVectors(r,n),Ft.subVectors(i,n),Nr.subVectors(e,n);const o=Lt.dot(Lt),a=Lt.dot(Ft),A=Lt.dot(Nr),l=Ft.dot(Ft),c=Ft.dot(Nr),d=o*l-a*a;if(d===0)return s.set(0,0,0),null;const f=1/d,p=(l*A-a*c)*f,E=(o*c-a*A)*f;return s.set(1-p-E,E,p)}static containsPoint(e,n,i,r){return this.getBarycoord(e,n,i,r,Wt)===null?!1:Wt.x>=0&&Wt.y>=0&&Wt.x+Wt.y<=1}static getUV(e,n,i,r,s,o,a,A){return Qi===!1&&(console.warn("THREE.Triangle.getUV() has been renamed to THREE.Triangle.getInterpolation()."),Qi=!0),this.getInterpolation(e,n,i,r,s,o,a,A)}static getInterpolation(e,n,i,r,s,o,a,A){return this.getBarycoord(e,n,i,r,Wt)===null?(A.x=0,A.y=0,"z"in A&&(A.z=0),"w"in A&&(A.w=0),null):(A.setScalar(0),A.addScaledVector(s,Wt.x),A.addScaledVector(o,Wt.y),A.addScaledVector(a,Wt.z),A)}static isFrontFacing(e,n,i,r){return Lt.subVectors(i,n),Ft.subVectors(e,n),Lt.cross(Ft).dot(r)<0}set(e,n,i){return this.a.copy(e),this.b.copy(n),this.c.copy(i),this}setFromPointsAndIndices(e,n,i,r){return this.a.copy(e[n]),this.b.copy(e[i]),this.c.copy(e[r]),this}setFromAttributeAndIndices(e,n,i,r){return this.a.fromBufferAttribute(e,n),this.b.fromBufferAttribute(e,i),this.c.fromBufferAttribute(e,r),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Lt.subVectors(this.c,this.b),Ft.subVectors(this.a,this.b),Lt.cross(Ft).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return Pn.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,n){return Pn.getBarycoord(e,this.a,this.b,this.c,n)}getUV(e,n,i,r,s){return Qi===!1&&(console.warn("THREE.Triangle.getUV() has been renamed to THREE.Triangle.getInterpolation()."),Qi=!0),Pn.getInterpolation(e,this.a,this.b,this.c,n,i,r,s)}getInterpolation(e,n,i,r,s){return Pn.getInterpolation(e,this.a,this.b,this.c,n,i,r,s)}containsPoint(e){return Pn.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return Pn.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,n){const i=this.a,r=this.b,s=this.c;let o,a;jn.subVectors(r,i),Un.subVectors(s,i),Gr.subVectors(e,i);const A=jn.dot(Gr),l=Un.dot(Gr);if(A<=0&&l<=0)return n.copy(i);zr.subVectors(e,r);const c=jn.dot(zr),d=Un.dot(zr);if(c>=0&&d<=c)return n.copy(r);const f=A*d-c*l;if(f<=0&&A>=0&&c<=0)return o=A/(A-c),n.copy(i).addScaledVector(jn,o);jr.subVectors(e,s);const p=jn.dot(jr),E=Un.dot(jr);if(E>=0&&p<=E)return n.copy(s);const g=p*l-A*E;if(g<=0&&l>=0&&E<=0)return a=l/(l-E),n.copy(i).addScaledVector(Un,a);const h=c*E-p*d;if(h<=0&&d-c>=0&&p-E>=0)return Pa.subVectors(s,r),a=(d-c)/(d-c+(p-E)),n.copy(r).addScaledVector(Pa,a);const u=1/(h+g+f);return o=g*u,a=f*u,n.copy(i).addScaledVector(jn,o).addScaledVector(Un,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}},ba={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},cn={h:0,s:0,l:0},Gi={h:0,s:0,l:0};function Ur(t,e,n){return n<0&&(n+=1),n>1&&(n-=1),n<.16666666666666666?t+(e-t)*6*n:n<.5?e:n<.6666666666666666?t+(e-t)*6*(.6666666666666666-n):t}var Ne=class{constructor(t,e,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(t,e,n)}set(t,e,n){if(e===void 0&&n===void 0){const i=t;i&&i.isColor?this.copy(i):typeof i=="number"?this.setHex(i):typeof i=="string"&&this.setStyle(i)}else this.setRGB(t,e,n);return this}setScalar(t){return this.r=t,this.g=t,this.b=t,this}setHex(t,e=ot){return t=Math.floor(t),this.r=(t>>16&255)/255,this.g=(t>>8&255)/255,this.b=(t&255)/255,je.toWorkingColorSpace(this,e),this}setRGB(t,e,n,i=je.workingColorSpace){return this.r=t,this.g=e,this.b=n,je.toWorkingColorSpace(this,i),this}setHSL(t,e,n,i=je.workingColorSpace){if(t=br(t,1),e=ft(e,0,1),n=ft(n,0,1),e===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+e):n+e-n*e,s=2*n-r;this.r=Ur(s,r,t+1/3),this.g=Ur(s,r,t),this.b=Ur(s,r,t-1/3)}return je.toWorkingColorSpace(this,i),this}setStyle(t,e=ot){function n(r){r!==void 0&&parseFloat(r)<1&&console.warn("THREE.Color: Alpha component of "+t+" will be ignored.")}let i;if(i=/^(\w+)\(([^\)]*)\)/.exec(t)){let r;const s=i[1],o=i[2];switch(s){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,e);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,e);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,e);break;default:console.warn("THREE.Color: Unknown color model "+t)}}else if(i=/^\#([A-Fa-f\d]+)$/.exec(t)){const r=i[1],s=r.length;if(s===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,e);if(s===6)return this.setHex(parseInt(r,16),e);console.warn("THREE.Color: Invalid hex color "+t)}else if(t&&t.length>0)return this.setColorName(t,e);return this}setColorName(t,e=ot){const n=ba[t.toLowerCase()];return n!==void 0?this.setHex(n,e):console.warn("THREE.Color: Unknown color "+t),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(t){return this.r=t.r,this.g=t.g,this.b=t.b,this}copySRGBToLinear(t){return this.r=Jn(t.r),this.g=Jn(t.g),this.b=Jn(t.b),this}copyLinearToSRGB(t){return this.r=_r(t.r),this.g=_r(t.g),this.b=_r(t.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(t=ot){return je.fromWorkingColorSpace(lt.copy(this),t),Math.round(ft(lt.r*255,0,255))*65536+Math.round(ft(lt.g*255,0,255))*256+Math.round(ft(lt.b*255,0,255))}getHexString(t=ot){return("000000"+this.getHex(t).toString(16)).slice(-6)}getHSL(t,e=je.workingColorSpace){je.fromWorkingColorSpace(lt.copy(this),e);const n=lt.r,i=lt.g,r=lt.b,s=Math.max(n,i,r),o=Math.min(n,i,r);let a,A;const l=(o+s)/2;if(o===s)a=0,A=0;else{const c=s-o;switch(A=l<=.5?c/(s+o):c/(2-s-o),s){case n:a=(i-r)/c+(i<r?6:0);break;case i:a=(r-n)/c+2;break;case r:a=(n-i)/c+4;break}a/=6}return t.h=a,t.s=A,t.l=l,t}getRGB(t,e=je.workingColorSpace){return je.fromWorkingColorSpace(lt.copy(this),e),t.r=lt.r,t.g=lt.g,t.b=lt.b,t}getStyle(t=ot){je.fromWorkingColorSpace(lt.copy(this),t);const e=lt.r,n=lt.g,i=lt.b;return t!==ot?`color(${t} ${e.toFixed(3)} ${n.toFixed(3)} ${i.toFixed(3)})`:`rgb(${Math.round(e*255)},${Math.round(n*255)},${Math.round(i*255)})`}offsetHSL(t,e,n){return this.getHSL(cn),this.setHSL(cn.h+t,cn.s+e,cn.l+n)}add(t){return this.r+=t.r,this.g+=t.g,this.b+=t.b,this}addColors(t,e){return this.r=t.r+e.r,this.g=t.g+e.g,this.b=t.b+e.b,this}addScalar(t){return this.r+=t,this.g+=t,this.b+=t,this}sub(t){return this.r=Math.max(0,this.r-t.r),this.g=Math.max(0,this.g-t.g),this.b=Math.max(0,this.b-t.b),this}multiply(t){return this.r*=t.r,this.g*=t.g,this.b*=t.b,this}multiplyScalar(t){return this.r*=t,this.g*=t,this.b*=t,this}lerp(t,e){return this.r+=(t.r-this.r)*e,this.g+=(t.g-this.g)*e,this.b+=(t.b-this.b)*e,this}lerpColors(t,e,n){return this.r=t.r+(e.r-t.r)*n,this.g=t.g+(e.g-t.g)*n,this.b=t.b+(e.b-t.b)*n,this}lerpHSL(t,e){this.getHSL(cn),t.getHSL(Gi);const n=oi(cn.h,Gi.h,e),i=oi(cn.s,Gi.s,e),r=oi(cn.l,Gi.l,e);return this.setHSL(n,i,r),this}setFromVector3(t){return this.r=t.x,this.g=t.y,this.b=t.z,this}applyMatrix3(t){const e=this.r,n=this.g,i=this.b,r=t.elements;return this.r=r[0]*e+r[3]*n+r[6]*i,this.g=r[1]*e+r[4]*n+r[7]*i,this.b=r[2]*e+r[5]*n+r[8]*i,this}equals(t){return t.r===this.r&&t.g===this.g&&t.b===this.b}fromArray(t,e=0){return this.r=t[e],this.g=t[e+1],this.b=t[e+2],this}toArray(t=[],e=0){return t[e]=this.r,t[e+1]=this.g,t[e+2]=this.b,t}fromBufferAttribute(t,e){return this.r=t.getX(e),this.g=t.getY(e),this.b=t.getZ(e),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},lt=new Ne;Ne.NAMES=ba;var gl=0,fi=class extends _n{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:gl++}),this.uuid=Ln(),this.name="",this.type="Material",this.blending=dn,this.side=$t,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=dr,this.blendDst=fr,this.blendEquation=fn,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Ne(0,0,0),this.blendAlpha=0,this.depthFunc=Si,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=sa,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Rn,this.stencilZFail=Rn,this.stencilZPass=Rn,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(t){this._alphaTest>0!=t>0&&this.version++,this._alphaTest=t}onBuild(){}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(t){if(t!==void 0)for(const e in t){const n=t[e];if(n===void 0){console.warn(`THREE.Material: parameter '${e}' has value of undefined.`);continue}const i=this[e];if(i===void 0){console.warn(`THREE.Material: '${e}' is not a property of THREE.${this.type}.`);continue}i&&i.isColor?i.set(n):i&&i.isVector3&&n&&n.isVector3?i.copy(n):this[e]=n}}toJSON(t){const e=t===void 0||typeof t=="string";e&&(t={textures:{},images:{}});const n={metadata:{version:4.6,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(t).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(t).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(t).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(t).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(t).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(t).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(t).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(t).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(t).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(t).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(t).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(t).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(t).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(t).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(t).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(t).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(t).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(t).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(t).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(t).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(t).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(t).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(t).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(t).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==dn&&(n.blending=this.blending),this.side!==$t&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==dr&&(n.blendSrc=this.blendSrc),this.blendDst!==fr&&(n.blendDst=this.blendDst),this.blendEquation!==fn&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==Si&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==sa&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Rn&&(n.stencilFail=this.stencilFail),this.stencilZFail!==Rn&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==Rn&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function i(r){const s=[];for(const o in r){const a=r[o];delete a.metadata,s.push(a)}return s}if(e){const r=i(t.textures),s=i(t.images);r.length>0&&(n.textures=r),s.length>0&&(n.images=s)}return n}clone(){return new this.constructor().copy(this)}copy(t){this.name=t.name,this.blending=t.blending,this.side=t.side,this.vertexColors=t.vertexColors,this.opacity=t.opacity,this.transparent=t.transparent,this.blendSrc=t.blendSrc,this.blendDst=t.blendDst,this.blendEquation=t.blendEquation,this.blendSrcAlpha=t.blendSrcAlpha,this.blendDstAlpha=t.blendDstAlpha,this.blendEquationAlpha=t.blendEquationAlpha,this.blendColor.copy(t.blendColor),this.blendAlpha=t.blendAlpha,this.depthFunc=t.depthFunc,this.depthTest=t.depthTest,this.depthWrite=t.depthWrite,this.stencilWriteMask=t.stencilWriteMask,this.stencilFunc=t.stencilFunc,this.stencilRef=t.stencilRef,this.stencilFuncMask=t.stencilFuncMask,this.stencilFail=t.stencilFail,this.stencilZFail=t.stencilZFail,this.stencilZPass=t.stencilZPass,this.stencilWrite=t.stencilWrite;const e=t.clippingPlanes;let n=null;if(e!==null){const i=e.length;n=new Array(i);for(let r=0;r!==i;++r)n[r]=e[r].clone()}return this.clippingPlanes=n,this.clipIntersection=t.clipIntersection,this.clipShadows=t.clipShadows,this.shadowSide=t.shadowSide,this.colorWrite=t.colorWrite,this.precision=t.precision,this.polygonOffset=t.polygonOffset,this.polygonOffsetFactor=t.polygonOffsetFactor,this.polygonOffsetUnits=t.polygonOffsetUnits,this.dithering=t.dithering,this.alphaTest=t.alphaTest,this.alphaHash=t.alphaHash,this.alphaToCoverage=t.alphaToCoverage,this.premultipliedAlpha=t.premultipliedAlpha,this.forceSinglePass=t.forceSinglePass,this.visible=t.visible,this.toneMapped=t.toneMapped,this.userData=JSON.parse(JSON.stringify(t.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(t){t===!0&&this.version++}},Ba=class extends fi{constructor(t){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new Ne(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.combine=vs,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.fog=t.fog,this}},et=new J,zi=new Fe,Ot=class{constructor(t,e,n=!1){if(Array.isArray(t))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,this.name="",this.array=t,this.itemSize=e,this.count=t!==void 0?t.length/e:0,this.normalized=n,this.usage=oa,this._updateRange={offset:0,count:-1},this.updateRanges=[],this.gpuType=sn,this.version=0}onUploadCallback(){}set needsUpdate(t){t===!0&&this.version++}get updateRange(){return console.warn("THREE.BufferAttribute: updateRange() is deprecated and will be removed in r169. Use addUpdateRange() instead."),this._updateRange}setUsage(t){return this.usage=t,this}addUpdateRange(t,e){this.updateRanges.push({start:t,count:e})}clearUpdateRanges(){this.updateRanges.length=0}copy(t){return this.name=t.name,this.array=new t.array.constructor(t.array),this.itemSize=t.itemSize,this.count=t.count,this.normalized=t.normalized,this.usage=t.usage,this.gpuType=t.gpuType,this}copyAt(t,e,n){t*=this.itemSize,n*=e.itemSize;for(let i=0,r=this.itemSize;i<r;i++)this.array[t+i]=e.array[n+i];return this}copyArray(t){return this.array.set(t),this}applyMatrix3(t){if(this.itemSize===2)for(let e=0,n=this.count;e<n;e++)zi.fromBufferAttribute(this,e),zi.applyMatrix3(t),this.setXY(e,zi.x,zi.y);else if(this.itemSize===3)for(let e=0,n=this.count;e<n;e++)et.fromBufferAttribute(this,e),et.applyMatrix3(t),this.setXYZ(e,et.x,et.y,et.z);return this}applyMatrix4(t){for(let e=0,n=this.count;e<n;e++)et.fromBufferAttribute(this,e),et.applyMatrix4(t),this.setXYZ(e,et.x,et.y,et.z);return this}applyNormalMatrix(t){for(let e=0,n=this.count;e<n;e++)et.fromBufferAttribute(this,e),et.applyNormalMatrix(t),this.setXYZ(e,et.x,et.y,et.z);return this}transformDirection(t){for(let e=0,n=this.count;e<n;e++)et.fromBufferAttribute(this,e),et.transformDirection(t),this.setXYZ(e,et.x,et.y,et.z);return this}set(t,e=0){return this.array.set(t,e),this}getComponent(t,e){let n=this.array[t*this.itemSize+e];return this.normalized&&(n=wn(n,this.array)),n}setComponent(t,e,n){return this.normalized&&(n=ht(n,this.array)),this.array[t*this.itemSize+e]=n,this}getX(t){let e=this.array[t*this.itemSize];return this.normalized&&(e=wn(e,this.array)),e}setX(t,e){return this.normalized&&(e=ht(e,this.array)),this.array[t*this.itemSize]=e,this}getY(t){let e=this.array[t*this.itemSize+1];return this.normalized&&(e=wn(e,this.array)),e}setY(t,e){return this.normalized&&(e=ht(e,this.array)),this.array[t*this.itemSize+1]=e,this}getZ(t){let e=this.array[t*this.itemSize+2];return this.normalized&&(e=wn(e,this.array)),e}setZ(t,e){return this.normalized&&(e=ht(e,this.array)),this.array[t*this.itemSize+2]=e,this}getW(t){let e=this.array[t*this.itemSize+3];return this.normalized&&(e=wn(e,this.array)),e}setW(t,e){return this.normalized&&(e=ht(e,this.array)),this.array[t*this.itemSize+3]=e,this}setXY(t,e,n){return t*=this.itemSize,this.normalized&&(e=ht(e,this.array),n=ht(n,this.array)),this.array[t+0]=e,this.array[t+1]=n,this}setXYZ(t,e,n,i){return t*=this.itemSize,this.normalized&&(e=ht(e,this.array),n=ht(n,this.array),i=ht(i,this.array)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=i,this}setXYZW(t,e,n,i,r){return t*=this.itemSize,this.normalized&&(e=ht(e,this.array),n=ht(n,this.array),i=ht(i,this.array),r=ht(r,this.array)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=i,this.array[t+3]=r,this}onUpload(t){return this.onUploadCallback=t,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const t={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(t.name=this.name),this.usage!==oa&&(t.usage=this.usage),t}},ka=class extends Ot{constructor(t,e,n){super(new Uint16Array(t),e,n)}},Ra=class extends Ot{constructor(t,e,n){super(new Uint32Array(t),e,n)}},mt=class extends Ot{constructor(t,e,n){super(new Float32Array(t),e,n)}},ml=0,Tt=new gt,Xr=new Jt,Xn=new J,Mt=new li,hi=new li,st=new J,un=class _o extends _n{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:ml++}),this.uuid=Ln(),this.name="",this.type="BufferGeometry",this.index=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(ca(e)?Ra:ka)(e,1):this.index=e,this}getAttribute(e){return this.attributes[e]}setAttribute(e,n){return this.attributes[e]=n,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,n,i=0){this.groups.push({start:e,count:n,materialIndex:i})}clearGroups(){this.groups=[]}setDrawRange(e,n){this.drawRange.start=e,this.drawRange.count=n}applyMatrix4(e){const n=this.attributes.position;n!==void 0&&(n.applyMatrix4(e),n.needsUpdate=!0);const i=this.attributes.normal;if(i!==void 0){const s=new Oe().getNormalMatrix(e);i.applyNormalMatrix(s),i.needsUpdate=!0}const r=this.attributes.tangent;return r!==void 0&&(r.transformDirection(e),r.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return Tt.makeRotationFromQuaternion(e),this.applyMatrix4(Tt),this}rotateX(e){return Tt.makeRotationX(e),this.applyMatrix4(Tt),this}rotateY(e){return Tt.makeRotationY(e),this.applyMatrix4(Tt),this}rotateZ(e){return Tt.makeRotationZ(e),this.applyMatrix4(Tt),this}translate(e,n,i){return Tt.makeTranslation(e,n,i),this.applyMatrix4(Tt),this}scale(e,n,i){return Tt.makeScale(e,n,i),this.applyMatrix4(Tt),this}lookAt(e){return Xr.lookAt(e),Xr.updateMatrix(),this.applyMatrix4(Xr.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Xn).negate(),this.translate(Xn.x,Xn.y,Xn.z),this}setFromPoints(e){const n=[];for(let i=0,r=e.length;i<r;i++){const s=e[i];n.push(s.x,s.y,s.z||0)}return this.setAttribute("position",new mt(n,3)),this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new li);const e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){console.error('THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box. Alternatively set "mesh.frustumCulled" to "false".',this),this.boundingBox.set(new J(-1/0,-1/0,-1/0),new J(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),n)for(let i=0,r=n.length;i<r;i++){const s=n[i];Mt.setFromBufferAttribute(s),this.morphTargetsRelative?(st.addVectors(this.boundingBox.min,Mt.min),this.boundingBox.expandByPoint(st),st.addVectors(this.boundingBox.max,Mt.max),this.boundingBox.expandByPoint(st)):(this.boundingBox.expandByPoint(Mt.min),this.boundingBox.expandByPoint(Mt.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new wi);const e=this.attributes.position,n=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){console.error('THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere. Alternatively set "mesh.frustumCulled" to "false".',this),this.boundingSphere.set(new J,1/0);return}if(e){const i=this.boundingSphere.center;if(Mt.setFromBufferAttribute(e),n)for(let s=0,o=n.length;s<o;s++){const a=n[s];hi.setFromBufferAttribute(a),this.morphTargetsRelative?(st.addVectors(Mt.min,hi.min),Mt.expandByPoint(st),st.addVectors(Mt.max,hi.max),Mt.expandByPoint(st)):(Mt.expandByPoint(hi.min),Mt.expandByPoint(hi.max))}Mt.getCenter(i);let r=0;for(let s=0,o=e.count;s<o;s++)st.fromBufferAttribute(e,s),r=Math.max(r,i.distanceToSquared(st));if(n)for(let s=0,o=n.length;s<o;s++){const a=n[s],A=this.morphTargetsRelative;for(let l=0,c=a.count;l<c;l++)st.fromBufferAttribute(a,l),A&&(Xn.fromBufferAttribute(e,l),st.add(Xn)),r=Math.max(r,i.distanceToSquared(st))}this.boundingSphere.radius=Math.sqrt(r),isNaN(this.boundingSphere.radius)&&console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,n=this.attributes;if(e===null||n.position===void 0||n.normal===void 0||n.uv===void 0){console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const i=e.array,r=n.position.array,s=n.normal.array,o=n.uv.array,a=r.length/3;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new Ot(new Float32Array(4*a),4));const A=this.getAttribute("tangent").array,l=[],c=[];for(let I=0;I<a;I++)l[I]=new J,c[I]=new J;const d=new J,f=new J,p=new J,E=new Fe,g=new Fe,h=new Fe,u=new J,v=new J;function y(I,H,W){d.fromArray(r,I*3),f.fromArray(r,H*3),p.fromArray(r,W*3),E.fromArray(o,I*2),g.fromArray(o,H*2),h.fromArray(o,W*2),f.sub(d),p.sub(d),g.sub(E),h.sub(E);const Y=1/(g.x*h.y-h.x*g.y);isFinite(Y)&&(u.copy(f).multiplyScalar(h.y).addScaledVector(p,-g.y).multiplyScalar(Y),v.copy(p).multiplyScalar(g.x).addScaledVector(f,-h.x).multiplyScalar(Y),l[I].add(u),l[H].add(u),l[W].add(u),c[I].add(v),c[H].add(v),c[W].add(v))}let T=this.groups;T.length===0&&(T=[{start:0,count:i.length}]);for(let I=0,H=T.length;I<H;++I){const W=T[I],Y=W.start,b=W.count;for(let Q=Y,G=Y+b;Q<G;Q+=3)y(i[Q+0],i[Q+1],i[Q+2])}const _=new J,C=new J,B=new J,z=new J;function M(I){B.fromArray(s,I*3),z.copy(B);const H=l[I];_.copy(H),_.sub(B.multiplyScalar(B.dot(H))).normalize(),C.crossVectors(z,H);const Y=C.dot(c[I])<0?-1:1;A[I*4]=_.x,A[I*4+1]=_.y,A[I*4+2]=_.z,A[I*4+3]=Y}for(let I=0,H=T.length;I<H;++I){const W=T[I],Y=W.start,b=W.count;for(let Q=Y,G=Y+b;Q<G;Q+=3)M(i[Q+0]),M(i[Q+1]),M(i[Q+2])}}computeVertexNormals(){const e=this.index,n=this.getAttribute("position");if(n!==void 0){let i=this.getAttribute("normal");if(i===void 0)i=new Ot(new Float32Array(n.count*3),3),this.setAttribute("normal",i);else for(let f=0,p=i.count;f<p;f++)i.setXYZ(f,0,0,0);const r=new J,s=new J,o=new J,a=new J,A=new J,l=new J,c=new J,d=new J;if(e)for(let f=0,p=e.count;f<p;f+=3){const E=e.getX(f+0),g=e.getX(f+1),h=e.getX(f+2);r.fromBufferAttribute(n,E),s.fromBufferAttribute(n,g),o.fromBufferAttribute(n,h),c.subVectors(o,s),d.subVectors(r,s),c.cross(d),a.fromBufferAttribute(i,E),A.fromBufferAttribute(i,g),l.fromBufferAttribute(i,h),a.add(c),A.add(c),l.add(c),i.setXYZ(E,a.x,a.y,a.z),i.setXYZ(g,A.x,A.y,A.z),i.setXYZ(h,l.x,l.y,l.z)}else for(let f=0,p=n.count;f<p;f+=3)r.fromBufferAttribute(n,f+0),s.fromBufferAttribute(n,f+1),o.fromBufferAttribute(n,f+2),c.subVectors(o,s),d.subVectors(r,s),c.cross(d),i.setXYZ(f+0,c.x,c.y,c.z),i.setXYZ(f+1,c.x,c.y,c.z),i.setXYZ(f+2,c.x,c.y,c.z);this.normalizeNormals(),i.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let n=0,i=e.count;n<i;n++)st.fromBufferAttribute(e,n),st.normalize(),e.setXYZ(n,st.x,st.y,st.z)}toNonIndexed(){function e(a,A){const l=a.array,c=a.itemSize,d=a.normalized,f=new l.constructor(A.length*c);let p=0,E=0;for(let g=0,h=A.length;g<h;g++){a.isInterleavedBufferAttribute?p=A[g]*a.data.stride+a.offset:p=A[g]*c;for(let u=0;u<c;u++)f[E++]=l[p++]}return new Ot(f,c,d)}if(this.index===null)return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const n=new _o,i=this.index.array,r=this.attributes;for(const a in r){const A=r[a],l=e(A,i);n.setAttribute(a,l)}const s=this.morphAttributes;for(const a in s){const A=[],l=s[a];for(let c=0,d=l.length;c<d;c++){const f=l[c],p=e(f,i);A.push(p)}n.morphAttributes[a]=A}n.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let a=0,A=o.length;a<A;a++){const l=o[a];n.addGroup(l.start,l.count,l.materialIndex)}return n}toJSON(){const e={metadata:{version:4.6,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const A=this.parameters;for(const l in A)A[l]!==void 0&&(e[l]=A[l]);return e}e.data={attributes:{}};const n=this.index;n!==null&&(e.data.index={type:n.array.constructor.name,array:Array.prototype.slice.call(n.array)});const i=this.attributes;for(const A in i){const l=i[A];e.data.attributes[A]=l.toJSON(e.data)}const r={};let s=!1;for(const A in this.morphAttributes){const l=this.morphAttributes[A],c=[];for(let d=0,f=l.length;d<f;d++){const p=l[d];c.push(p.toJSON(e.data))}c.length>0&&(r[A]=c,s=!0)}s&&(e.data.morphAttributes=r,e.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));const a=this.boundingSphere;return a!==null&&(e.data.boundingSphere={center:a.center.toArray(),radius:a.radius}),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const n={};this.name=e.name;const i=e.index;i!==null&&this.setIndex(i.clone(n));const r=e.attributes;for(const l in r){const c=r[l];this.setAttribute(l,c.clone(n))}const s=e.morphAttributes;for(const l in s){const c=[],d=s[l];for(let f=0,p=d.length;f<p;f++)c.push(d[f].clone(n));this.morphAttributes[l]=c}this.morphTargetsRelative=e.morphTargetsRelative;const o=e.groups;for(let l=0,c=o.length;l<c;l++){const d=o[l];this.addGroup(d.start,d.count,d.materialIndex)}const a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());const A=e.boundingSphere;return A!==null&&(this.boundingSphere=A.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}},_a=new gt,vn=new Ea,ji=new wi,La=new J,Zn=new J,Fn=new J,Wn=new J,Zr=new J,Ui=new J,Xi=new Fe,Zi=new Fe,Fi=new Fe,wa=new J,Ja=new J,Oa=new J,Wi=new J,Yi=new J,Yt=class extends Jt{constructor(t=new un,e=new Ba){super(),this.isMesh=!0,this.type="Mesh",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),t.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=t.morphTargetInfluences.slice()),t.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},t.morphTargetDictionary)),this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}updateMorphTargets(){const e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){const i=e[n[0]];if(i!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,s=i.length;r<s;r++){const o=i[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}getVertexPosition(t,e){const n=this.geometry,i=n.attributes.position,r=n.morphAttributes.position,s=n.morphTargetsRelative;e.fromBufferAttribute(i,t);const o=this.morphTargetInfluences;if(r&&o){Ui.set(0,0,0);for(let a=0,A=r.length;a<A;a++){const l=o[a],c=r[a];l!==0&&(Zr.fromBufferAttribute(c,t),s?Ui.addScaledVector(Zr,l):Ui.addScaledVector(Zr.sub(e),l))}e.add(Ui)}return e}raycast(t,e){const n=this.geometry,i=this.material,r=this.matrixWorld;i!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),ji.copy(n.boundingSphere),ji.applyMatrix4(r),vn.copy(t.ray).recast(t.near),!(ji.containsPoint(vn.origin)===!1&&(vn.intersectSphere(ji,La)===null||vn.origin.distanceToSquared(La)>(t.far-t.near)**2))&&(_a.copy(r).invert(),vn.copy(t.ray).applyMatrix4(_a),!(n.boundingBox!==null&&vn.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(t,e,vn)))}_computeIntersections(t,e,n){let i;const r=this.geometry,s=this.material,o=r.index,a=r.attributes.position,A=r.attributes.uv,l=r.attributes.uv1,c=r.attributes.normal,d=r.groups,f=r.drawRange;if(o!==null)if(Array.isArray(s))for(let p=0,E=d.length;p<E;p++){const g=d[p],h=s[g.materialIndex],u=Math.max(g.start,f.start),v=Math.min(o.count,Math.min(g.start+g.count,f.start+f.count));for(let y=u,T=v;y<T;y+=3){const _=o.getX(y),C=o.getX(y+1),B=o.getX(y+2);i=Vi(this,h,t,n,A,l,c,_,C,B),i&&(i.faceIndex=Math.floor(y/3),i.face.materialIndex=g.materialIndex,e.push(i))}}else{const p=Math.max(0,f.start),E=Math.min(o.count,f.start+f.count);for(let g=p,h=E;g<h;g+=3){const u=o.getX(g),v=o.getX(g+1),y=o.getX(g+2);i=Vi(this,s,t,n,A,l,c,u,v,y),i&&(i.faceIndex=Math.floor(g/3),e.push(i))}}else if(a!==void 0)if(Array.isArray(s))for(let p=0,E=d.length;p<E;p++){const g=d[p],h=s[g.materialIndex],u=Math.max(g.start,f.start),v=Math.min(a.count,Math.min(g.start+g.count,f.start+f.count));for(let y=u,T=v;y<T;y+=3){const _=y,C=y+1,B=y+2;i=Vi(this,h,t,n,A,l,c,_,C,B),i&&(i.faceIndex=Math.floor(y/3),i.face.materialIndex=g.materialIndex,e.push(i))}}else{const p=Math.max(0,f.start),E=Math.min(a.count,f.start+f.count);for(let g=p,h=E;g<h;g+=3){const u=g,v=g+1,y=g+2;i=Vi(this,s,t,n,A,l,c,u,v,y),i&&(i.faceIndex=Math.floor(g/3),e.push(i))}}}};function El(t,e,n,i,r,s,o,a){let A;if(e.side===ut?A=i.intersectTriangle(o,s,r,!0,a):A=i.intersectTriangle(r,s,o,e.side===$t,a),A===null)return null;Yi.copy(a),Yi.applyMatrix4(t.matrixWorld);const l=n.ray.origin.distanceTo(Yi);return l<n.near||l>n.far?null:{distance:l,point:Yi.clone(),object:t}}function Vi(t,e,n,i,r,s,o,a,A,l){t.getVertexPosition(a,Zn),t.getVertexPosition(A,Fn),t.getVertexPosition(l,Wn);const c=El(t,e,n,i,Zn,Fn,Wn,Wi);if(c){r&&(Xi.fromBufferAttribute(r,a),Zi.fromBufferAttribute(r,A),Fi.fromBufferAttribute(r,l),c.uv=Ni.getInterpolation(Wi,Zn,Fn,Wn,Xi,Zi,Fi,new Fe)),s&&(Xi.fromBufferAttribute(s,a),Zi.fromBufferAttribute(s,A),Fi.fromBufferAttribute(s,l),c.uv1=Ni.getInterpolation(Wi,Zn,Fn,Wn,Xi,Zi,Fi,new Fe),c.uv2=c.uv1),o&&(wa.fromBufferAttribute(o,a),Ja.fromBufferAttribute(o,A),Oa.fromBufferAttribute(o,l),c.normal=Ni.getInterpolation(Wi,Zn,Fn,Wn,wa,Ja,Oa,new J),c.normal.dot(i.direction)>0&&c.normal.multiplyScalar(-1));const d={a,b:A,c:l,normal:new J,materialIndex:0};Ni.getNormal(Zn,Fn,Wn,d.normal),c.face=d}return c}var Fr=class Lo extends un{constructor(e=1,n=1,i=1,r=1,s=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:n,depth:i,widthSegments:r,heightSegments:s,depthSegments:o};const a=this;r=Math.floor(r),s=Math.floor(s),o=Math.floor(o);const A=[],l=[],c=[],d=[];let f=0,p=0;E("z","y","x",-1,-1,i,n,e,o,s,0),E("z","y","x",1,-1,i,n,-e,o,s,1),E("x","z","y",1,1,e,i,n,r,o,2),E("x","z","y",1,-1,e,i,-n,r,o,3),E("x","y","z",1,-1,e,n,i,r,s,4),E("x","y","z",-1,-1,e,n,-i,r,s,5),this.setIndex(A),this.setAttribute("position",new mt(l,3)),this.setAttribute("normal",new mt(c,3)),this.setAttribute("uv",new mt(d,2));function E(g,h,u,v,y,T,_,C,B,z,M){const I=T/B,H=_/z,W=T/2,Y=_/2,b=C/2,Q=B+1,G=z+1;let q=0,U=0;const j=new J;for(let X=0;X<G;X++){const ee=X*H-Y;for(let k=0;k<Q;k++){const Z=k*I-W;j[g]=Z*v,j[h]=ee*y,j[u]=b,l.push(j.x,j.y,j.z),j[g]=0,j[h]=0,j[u]=C>0?1:-1,c.push(j.x,j.y,j.z),d.push(k/B),d.push(1-X/z),q+=1}}for(let X=0;X<z;X++)for(let ee=0;ee<B;ee++){const k=f+ee+Q*X,Z=f+ee+Q*(X+1),re=f+(ee+1)+Q*(X+1),se=f+(ee+1)+Q*X;A.push(k,Z,se),A.push(Z,re,se),U+=6}a.addGroup(p,U,M),p+=U,f+=q}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Lo(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}};function Yn(t){const e={};for(const n in t){e[n]={};for(const i in t[n]){const r=t[n][i];r&&(r.isColor||r.isMatrix3||r.isMatrix4||r.isVector2||r.isVector3||r.isVector4||r.isTexture||r.isQuaternion)?r.isRenderTargetTexture?(console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[n][i]=null):e[n][i]=r.clone():Array.isArray(r)?e[n][i]=r.slice():e[n][i]=r}}return e}function Et(t){const e={};for(let n=0;n<t.length;n++){const i=Yn(t[n]);for(const r in i)e[r]=i[r]}return e}function Sl(t){const e=[];for(let n=0;n<t.length;n++)e.push(t[n].clone());return e}function Da(t){return t.getRenderTarget()===null?t.outputColorSpace:je.workingColorSpace}var vl={clone:Yn,merge:Et},xl=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,Ml=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,Vt=class extends fi{constructor(t){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=xl,this.fragmentShader=Ml,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={derivatives:!1,fragDepth:!1,drawBuffers:!1,shaderTextureLOD:!1,clipCullDistance:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,t!==void 0&&this.setValues(t)}copy(t){return super.copy(t),this.fragmentShader=t.fragmentShader,this.vertexShader=t.vertexShader,this.uniforms=Yn(t.uniforms),this.uniformsGroups=Sl(t.uniformsGroups),this.defines=Object.assign({},t.defines),this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.fog=t.fog,this.lights=t.lights,this.clipping=t.clipping,this.extensions=Object.assign({},t.extensions),this.glslVersion=t.glslVersion,this}toJSON(t){const e=super.toJSON(t);e.glslVersion=this.glslVersion,e.uniforms={};for(const i in this.uniforms){const s=this.uniforms[i].value;s&&s.isTexture?e.uniforms[i]={type:"t",value:s.toJSON(t).uuid}:s&&s.isColor?e.uniforms[i]={type:"c",value:s.getHex()}:s&&s.isVector2?e.uniforms[i]={type:"v2",value:s.toArray()}:s&&s.isVector3?e.uniforms[i]={type:"v3",value:s.toArray()}:s&&s.isVector4?e.uniforms[i]={type:"v4",value:s.toArray()}:s&&s.isMatrix3?e.uniforms[i]={type:"m3",value:s.toArray()}:s&&s.isMatrix4?e.uniforms[i]={type:"m4",value:s.toArray()}:e.uniforms[i]={value:s}}Object.keys(this.defines).length>0&&(e.defines=this.defines),e.vertexShader=this.vertexShader,e.fragmentShader=this.fragmentShader,e.lights=this.lights,e.clipping=this.clipping;const n={};for(const i in this.extensions)this.extensions[i]===!0&&(n[i]=!0);return Object.keys(n).length>0&&(e.extensions=n),e}},Ha=class extends Jt{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new gt,this.projectionMatrix=new gt,this.projectionMatrixInverse=new gt,this.coordinateSystem=jt}copy(t,e){return super.copy(t,e),this.matrixWorldInverse.copy(t.matrixWorldInverse),this.projectionMatrix.copy(t.projectionMatrix),this.projectionMatrixInverse.copy(t.projectionMatrixInverse),this.coordinateSystem=t.coordinateSystem,this}getWorldDirection(t){return super.getWorldDirection(t).negate()}updateMatrixWorld(t){super.updateMatrixWorld(t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(t,e){super.updateWorldMatrix(t,e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}},Pt=class extends Ha{constructor(t=50,e=1,n=.1,i=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=t,this.zoom=1,this.near=n,this.far=i,this.focus=10,this.aspect=e,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.fov=t.fov,this.zoom=t.zoom,this.near=t.near,this.far=t.far,this.focus=t.focus,this.aspect=t.aspect,this.view=t.view===null?null:Object.assign({},t.view),this.filmGauge=t.filmGauge,this.filmOffset=t.filmOffset,this}setFocalLength(t){const e=.5*this.getFilmHeight()/t;this.fov=ai*2*Math.atan(e),this.updateProjectionMatrix()}getFocalLength(){const t=Math.tan(si*.5*this.fov);return .5*this.getFilmHeight()/t}getEffectiveFOV(){return ai*2*Math.atan(Math.tan(si*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}setViewOffset(t,e,n,i,r,s){this.aspect=t/e,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=i,this.view.width=r,this.view.height=s,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=this.near;let e=t*Math.tan(si*.5*this.fov)/this.zoom,n=2*e,i=this.aspect*n,r=-.5*i;const s=this.view;if(this.view!==null&&this.view.enabled){const a=s.fullWidth,A=s.fullHeight;r+=s.offsetX*i/a,e-=s.offsetY*n/A,i*=s.width/a,n*=s.height/A}const o=this.filmOffset;o!==0&&(r+=t*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+i,e,e-n,t,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.fov=this.fov,e.object.zoom=this.zoom,e.object.near=this.near,e.object.far=this.far,e.object.focus=this.focus,e.object.aspect=this.aspect,this.view!==null&&(e.object.view=Object.assign({},this.view)),e.object.filmGauge=this.filmGauge,e.object.filmOffset=this.filmOffset,e}},Vn=-90,qn=1,Il=class extends Jt{constructor(t,e,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const i=new Pt(Vn,qn,t,e);i.layers=this.layers,this.add(i);const r=new Pt(Vn,qn,t,e);r.layers=this.layers,this.add(r);const s=new Pt(Vn,qn,t,e);s.layers=this.layers,this.add(s);const o=new Pt(Vn,qn,t,e);o.layers=this.layers,this.add(o);const a=new Pt(Vn,qn,t,e);a.layers=this.layers,this.add(a);const A=new Pt(Vn,qn,t,e);A.layers=this.layers,this.add(A)}updateCoordinateSystem(){const t=this.coordinateSystem,e=this.children.concat(),[n,i,r,s,o,a]=e;for(const A of e)this.remove(A);if(t===jt)n.up.set(0,1,0),n.lookAt(1,0,0),i.up.set(0,1,0),i.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),s.up.set(0,0,1),s.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),a.up.set(0,1,0),a.lookAt(0,0,-1);else if(t===Pi)n.up.set(0,-1,0),n.lookAt(-1,0,0),i.up.set(0,-1,0),i.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),s.up.set(0,0,-1),s.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),a.up.set(0,-1,0),a.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+t);for(const A of e)this.add(A),A.updateMatrixWorld()}update(t,e){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:i}=this;this.coordinateSystem!==t.coordinateSystem&&(this.coordinateSystem=t.coordinateSystem,this.updateCoordinateSystem());const[r,s,o,a,A,l]=this.children,c=t.getRenderTarget(),d=t.getActiveCubeFace(),f=t.getActiveMipmapLevel(),p=t.xr.enabled;t.xr.enabled=!1;const E=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,t.setRenderTarget(n,0,i),t.render(e,r),t.setRenderTarget(n,1,i),t.render(e,s),t.setRenderTarget(n,2,i),t.render(e,o),t.setRenderTarget(n,3,i),t.render(e,a),t.setRenderTarget(n,4,i),t.render(e,A),n.texture.generateMipmaps=E,t.setRenderTarget(n,5,i),t.render(e,l),t.setRenderTarget(c,d,f),t.xr.enabled=p,n.texture.needsPMREMUpdate=!0}},Qa=class extends wt{constructor(t,e,n,i,r,s,o,a,A,l){t=t!==void 0?t:[],e=e!==void 0?e:bn,super(t,e,n,i,r,s,o,a,A,l),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(t){this.image=t}},yl=class extends mn{constructor(t=1,e={}){super(t,t,e),this.isWebGLCubeRenderTarget=!0;const n={width:t,height:t,depth:1},i=[n,n,n,n,n,n];e.encoding!==void 0&&(Ai("THREE.WebGLCubeRenderTarget: option.encoding has been replaced by option.colorSpace."),e.colorSpace=e.encoding===gn?ot:Ct),this.texture=new Qa(i,e.mapping,e.wrapS,e.wrapT,e.magFilter,e.minFilter,e.format,e.type,e.anisotropy,e.colorSpace),this.texture.isRenderTargetTexture=!0,this.texture.generateMipmaps=e.generateMipmaps!==void 0?e.generateMipmaps:!1,this.texture.minFilter=e.minFilter!==void 0?e.minFilter:yt}fromEquirectangularTexture(t,e){this.texture.type=e.type,this.texture.colorSpace=e.colorSpace,this.texture.generateMipmaps=e.generateMipmaps,this.texture.minFilter=e.minFilter,this.texture.magFilter=e.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},i=new Fr(5,5,5),r=new Vt({name:"CubemapFromEquirect",uniforms:Yn(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:ut,blending:en});r.uniforms.tEquirect.value=e;const s=new Yt(i,r),o=e.minFilter;return e.minFilter===ii&&(e.minFilter=yt),new Il(1,10,this).update(t,s),e.minFilter=o,s.geometry.dispose(),s.material.dispose(),this}clear(t,e,n,i){const r=t.getRenderTarget();for(let s=0;s<6;s++)t.setRenderTarget(this,s),t.clear(e,n,i);t.setRenderTarget(r)}},Wr=new J,Cl=new J,Tl=new Oe,xn=class{constructor(t=new J(1,0,0),e=0){this.isPlane=!0,this.normal=t,this.constant=e}set(t,e){return this.normal.copy(t),this.constant=e,this}setComponents(t,e,n,i){return this.normal.set(t,e,n),this.constant=i,this}setFromNormalAndCoplanarPoint(t,e){return this.normal.copy(t),this.constant=-e.dot(this.normal),this}setFromCoplanarPoints(t,e,n){const i=Wr.subVectors(n,e).cross(Cl.subVectors(t,e)).normalize();return this.setFromNormalAndCoplanarPoint(i,t),this}copy(t){return this.normal.copy(t.normal),this.constant=t.constant,this}normalize(){const t=1/this.normal.length();return this.normal.multiplyScalar(t),this.constant*=t,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(t){return this.normal.dot(t)+this.constant}distanceToSphere(t){return this.distanceToPoint(t.center)-t.radius}projectPoint(t,e){return e.copy(t).addScaledVector(this.normal,-this.distanceToPoint(t))}intersectLine(t,e){const n=t.delta(Wr),i=this.normal.dot(n);if(i===0)return this.distanceToPoint(t.start)===0?e.copy(t.start):null;const r=-(t.start.dot(this.normal)+this.constant)/i;return r<0||r>1?null:e.copy(t.start).addScaledVector(n,r)}intersectsLine(t){const e=this.distanceToPoint(t.start),n=this.distanceToPoint(t.end);return e<0&&n>0||n<0&&e>0}intersectsBox(t){return t.intersectsPlane(this)}intersectsSphere(t){return t.intersectsPlane(this)}coplanarPoint(t){return t.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(t,e){const n=e||Tl.getNormalMatrix(t),i=this.coplanarPoint(Wr).applyMatrix4(t),r=this.normal.applyMatrix3(n).normalize();return this.constant=-i.dot(r),this}translate(t){return this.constant-=t.dot(this.normal),this}equals(t){return t.normal.equals(this.normal)&&t.constant===this.constant}clone(){return new this.constructor().copy(this)}},Mn=new wi,qi=new J,Na=class{constructor(t=new xn,e=new xn,n=new xn,i=new xn,r=new xn,s=new xn){this.planes=[t,e,n,i,r,s]}set(t,e,n,i,r,s){const o=this.planes;return o[0].copy(t),o[1].copy(e),o[2].copy(n),o[3].copy(i),o[4].copy(r),o[5].copy(s),this}copy(t){const e=this.planes;for(let n=0;n<6;n++)e[n].copy(t.planes[n]);return this}setFromProjectionMatrix(t,e=jt){const n=this.planes,i=t.elements,r=i[0],s=i[1],o=i[2],a=i[3],A=i[4],l=i[5],c=i[6],d=i[7],f=i[8],p=i[9],E=i[10],g=i[11],h=i[12],u=i[13],v=i[14],y=i[15];if(n[0].setComponents(a-r,d-A,g-f,y-h).normalize(),n[1].setComponents(a+r,d+A,g+f,y+h).normalize(),n[2].setComponents(a+s,d+l,g+p,y+u).normalize(),n[3].setComponents(a-s,d-l,g-p,y-u).normalize(),n[4].setComponents(a-o,d-c,g-E,y-v).normalize(),e===jt)n[5].setComponents(a+o,d+c,g+E,y+v).normalize();else if(e===Pi)n[5].setComponents(o,c,E,v).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+e);return this}intersectsObject(t){if(t.boundingSphere!==void 0)t.boundingSphere===null&&t.computeBoundingSphere(),Mn.copy(t.boundingSphere).applyMatrix4(t.matrixWorld);else{const e=t.geometry;e.boundingSphere===null&&e.computeBoundingSphere(),Mn.copy(e.boundingSphere).applyMatrix4(t.matrixWorld)}return this.intersectsSphere(Mn)}intersectsSprite(t){return Mn.center.set(0,0,0),Mn.radius=.7071067811865476,Mn.applyMatrix4(t.matrixWorld),this.intersectsSphere(Mn)}intersectsSphere(t){const e=this.planes,n=t.center,i=-t.radius;for(let r=0;r<6;r++)if(e[r].distanceToPoint(n)<i)return!1;return!0}intersectsBox(t){const e=this.planes;for(let n=0;n<6;n++){const i=e[n];if(qi.x=i.normal.x>0?t.max.x:t.min.x,qi.y=i.normal.y>0?t.max.y:t.min.y,qi.z=i.normal.z>0?t.max.z:t.min.z,i.distanceToPoint(qi)<0)return!1}return!0}containsPoint(t){const e=this.planes;for(let n=0;n<6;n++)if(e[n].distanceToPoint(t)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}};function Ga(){let t=null,e=!1,n=null,i=null;function r(s,o){n(s,o),i=t.requestAnimationFrame(r)}return{start:function(){e!==!0&&n!==null&&(i=t.requestAnimationFrame(r),e=!0)},stop:function(){t.cancelAnimationFrame(i),e=!1},setAnimationLoop:function(s){n=s},setContext:function(s){t=s}}}function Pl(t,e){const n=e.isWebGL2,i=new WeakMap;function r(l,c){const d=l.array,f=l.usage,p=d.byteLength,E=t.createBuffer();t.bindBuffer(c,E),t.bufferData(c,d,f),l.onUploadCallback();let g;if(d instanceof Float32Array)g=t.FLOAT;else if(d instanceof Uint16Array)if(l.isFloat16BufferAttribute)if(n)g=t.HALF_FLOAT;else throw new Error("THREE.WebGLAttributes: Usage of Float16BufferAttribute requires WebGL2.");else g=t.UNSIGNED_SHORT;else if(d instanceof Int16Array)g=t.SHORT;else if(d instanceof Uint32Array)g=t.UNSIGNED_INT;else if(d instanceof Int32Array)g=t.INT;else if(d instanceof Int8Array)g=t.BYTE;else if(d instanceof Uint8Array)g=t.UNSIGNED_BYTE;else if(d instanceof Uint8ClampedArray)g=t.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+d);return{buffer:E,type:g,bytesPerElement:d.BYTES_PER_ELEMENT,version:l.version,size:p}}function s(l,c,d){const f=c.array,p=c._updateRange,E=c.updateRanges;if(t.bindBuffer(d,l),p.count===-1&&E.length===0&&t.bufferSubData(d,0,f),E.length!==0){for(let g=0,h=E.length;g<h;g++){const u=E[g];n?t.bufferSubData(d,u.start*f.BYTES_PER_ELEMENT,f,u.start,u.count):t.bufferSubData(d,u.start*f.BYTES_PER_ELEMENT,f.subarray(u.start,u.start+u.count))}c.clearUpdateRanges()}p.count!==-1&&(n?t.bufferSubData(d,p.offset*f.BYTES_PER_ELEMENT,f,p.offset,p.count):t.bufferSubData(d,p.offset*f.BYTES_PER_ELEMENT,f.subarray(p.offset,p.offset+p.count)),p.count=-1),c.onUploadCallback()}function o(l){return l.isInterleavedBufferAttribute&&(l=l.data),i.get(l)}function a(l){l.isInterleavedBufferAttribute&&(l=l.data);const c=i.get(l);c&&(t.deleteBuffer(c.buffer),i.delete(l))}function A(l,c){if(l.isGLBufferAttribute){const f=i.get(l);(!f||f.version<l.version)&&i.set(l,{buffer:l.buffer,type:l.type,bytesPerElement:l.elementSize,version:l.version});return}l.isInterleavedBufferAttribute&&(l=l.data);const d=i.get(l);if(d===void 0)i.set(l,r(l,c));else if(d.version<l.version){if(d.size!==l.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");s(d.buffer,l,c),d.version=l.version}}return{get:o,remove:a,update:A}}var bl=class wo extends un{constructor(e=1,n=1,i=1,r=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:n,widthSegments:i,heightSegments:r};const s=e/2,o=n/2,a=Math.floor(i),A=Math.floor(r),l=a+1,c=A+1,d=e/a,f=n/A,p=[],E=[],g=[],h=[];for(let u=0;u<c;u++){const v=u*f-o;for(let y=0;y<l;y++){const T=y*d-s;E.push(T,-v,0),g.push(0,0,1),h.push(y/a),h.push(1-u/A)}}for(let u=0;u<A;u++)for(let v=0;v<a;v++){const y=v+l*u,T=v+l*(u+1),_=v+1+l*(u+1),C=v+1+l*u;p.push(y,T,C),p.push(T,_,C)}this.setIndex(p),this.setAttribute("position",new mt(E,3)),this.setAttribute("normal",new mt(g,3)),this.setAttribute("uv",new mt(h,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new wo(e.width,e.height,e.widthSegments,e.heightSegments)}},Bl=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,kl=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,Rl=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,_l=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Ll=`#ifdef USE_ALPHATEST
	if ( diffuseColor.a < alphaTest ) discard;
#endif`,wl=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Jl=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Ol=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Dl=`#ifdef USE_BATCHING
	attribute float batchId;
	uniform highp sampler2D batchingTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,Hl=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( batchId );
#endif`,Ql=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Nl=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Gl=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,zl=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,jl=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Ul=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#pragma unroll_loop_start
	for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
		plane = clippingPlanes[ i ];
		if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
	}
	#pragma unroll_loop_end
	#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
		bool clipped = true;
		#pragma unroll_loop_start
		for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
		}
		#pragma unroll_loop_end
		if ( clipped ) discard;
	#endif
#endif`,Xl=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Zl=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,Fl=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Wl=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,Yl=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,Vl=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR )
	varying vec3 vColor;
#endif`,ql=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif`,Kl=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
float luminance( const in vec3 rgb ) {
	const vec3 weights = vec3( 0.2126729, 0.7151522, 0.0721750 );
	return dot( weights, rgb );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,$l=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,ec=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,tc=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,nc=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,ic=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,rc=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,sc="gl_FragColor = linearToOutputTexel( gl_FragColor );",ac=`
const mat3 LINEAR_SRGB_TO_LINEAR_DISPLAY_P3 = mat3(
	vec3( 0.8224621, 0.177538, 0.0 ),
	vec3( 0.0331941, 0.9668058, 0.0 ),
	vec3( 0.0170827, 0.0723974, 0.9105199 )
);
const mat3 LINEAR_DISPLAY_P3_TO_LINEAR_SRGB = mat3(
	vec3( 1.2249401, - 0.2249404, 0.0 ),
	vec3( - 0.0420569, 1.0420571, 0.0 ),
	vec3( - 0.0196376, - 0.0786361, 1.0982735 )
);
vec4 LinearSRGBToLinearDisplayP3( in vec4 value ) {
	return vec4( value.rgb * LINEAR_SRGB_TO_LINEAR_DISPLAY_P3, value.a );
}
vec4 LinearDisplayP3ToLinearSRGB( in vec4 value ) {
	return vec4( value.rgb * LINEAR_DISPLAY_P3_TO_LINEAR_SRGB, value.a );
}
vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}
vec4 LinearToLinear( in vec4 value ) {
	return value;
}
vec4 LinearTosRGB( in vec4 value ) {
	return sRGBTransferOETF( value );
}`,oc=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,Ac=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
	
#endif`,lc=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,cc=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,uc=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,dc=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,fc=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,hc=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,pc=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,gc=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,mc=`#ifdef USE_LIGHTMAP
	vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
	vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
	reflectedLight.indirectDiffuse += lightMapIrradiance;
#endif`,Ec=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,Sc=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,vc=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,xc=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	#if defined ( LEGACY_LIGHTS )
		if ( cutoffDistance > 0.0 && decayExponent > 0.0 ) {
			return pow( saturate( - lightDistance / cutoffDistance + 1.0 ), decayExponent );
		}
		return 1.0;
	#else
		float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
		if ( cutoffDistance > 0.0 ) {
			distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
		}
		return distanceFalloff;
	#endif
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,Mc=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,Ic=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,yc=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,Cc=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Tc=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,Pc=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,bc=`struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,Bc=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,kc=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,Rc=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,_c=`#if defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT )
	gl_FragDepthEXT = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Lc=`#if defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,wc=`#ifdef USE_LOGDEPTHBUF
	#ifdef USE_LOGDEPTHBUF_EXT
		varying float vFragDepth;
		varying float vIsPerspective;
	#else
		uniform float logDepthBufFC;
	#endif
#endif`,Jc=`#ifdef USE_LOGDEPTHBUF
	#ifdef USE_LOGDEPTHBUF_EXT
		vFragDepth = 1.0 + gl_Position.w;
		vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
	#else
		if ( isPerspectiveMatrix( projectionMatrix ) ) {
			gl_Position.z = log2( max( EPSILON, gl_Position.w + 1.0 ) ) * logDepthBufFC - 1.0;
			gl_Position.z *= gl_Position.w;
		}
	#endif
#endif`,Oc=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = vec4( mix( pow( sampledDiffuseColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), sampledDiffuseColor.rgb * 0.0773993808, vec3( lessThanEqual( sampledDiffuseColor.rgb, vec3( 0.04045 ) ) ) ), sampledDiffuseColor.w );
	
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Dc=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Hc=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,Qc=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Nc=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Gc=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,zc=`#if defined( USE_MORPHCOLORS ) && defined( MORPHTARGETS_TEXTURE )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,jc=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	#ifdef MORPHTARGETS_TEXTURE
		for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
			if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
		}
	#else
		objectNormal += morphNormal0 * morphTargetInfluences[ 0 ];
		objectNormal += morphNormal1 * morphTargetInfluences[ 1 ];
		objectNormal += morphNormal2 * morphTargetInfluences[ 2 ];
		objectNormal += morphNormal3 * morphTargetInfluences[ 3 ];
	#endif
#endif`,Uc=`#ifdef USE_MORPHTARGETS
	uniform float morphTargetBaseInfluence;
	#ifdef MORPHTARGETS_TEXTURE
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
		uniform sampler2DArray morphTargetsTexture;
		uniform ivec2 morphTargetsTextureSize;
		vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
			int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
			int y = texelIndex / morphTargetsTextureSize.x;
			int x = texelIndex - y * morphTargetsTextureSize.x;
			ivec3 morphUV = ivec3( x, y, morphTargetIndex );
			return texelFetch( morphTargetsTexture, morphUV, 0 );
		}
	#else
		#ifndef USE_MORPHNORMALS
			uniform float morphTargetInfluences[ 8 ];
		#else
			uniform float morphTargetInfluences[ 4 ];
		#endif
	#endif
#endif`,Xc=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	#ifdef MORPHTARGETS_TEXTURE
		for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
			if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
		}
	#else
		transformed += morphTarget0 * morphTargetInfluences[ 0 ];
		transformed += morphTarget1 * morphTargetInfluences[ 1 ];
		transformed += morphTarget2 * morphTargetInfluences[ 2 ];
		transformed += morphTarget3 * morphTargetInfluences[ 3 ];
		#ifndef USE_MORPHNORMALS
			transformed += morphTarget4 * morphTargetInfluences[ 4 ];
			transformed += morphTarget5 * morphTargetInfluences[ 5 ];
			transformed += morphTarget6 * morphTargetInfluences[ 6 ];
			transformed += morphTarget7 * morphTargetInfluences[ 7 ];
		#endif
	#endif
#endif`,Zc=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,Fc=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,Wc=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Yc=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Vc=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,qc=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,Kc=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,$c=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,eu=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,tu=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,nu=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,iu=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;
const vec3 PackFactors = vec3( 256. * 256. * 256., 256. * 256., 256. );
const vec4 UnpackFactors = UnpackDownscale / vec4( PackFactors, 1. );
const float ShiftRight8 = 1. / 256.;
vec4 packDepthToRGBA( const in float v ) {
	vec4 r = vec4( fract( v * PackFactors ), v );
	r.yzw -= r.xyz * ShiftRight8;	return r * PackUpscale;
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors );
}
vec2 packDepthToRG( in highp float v ) {
	return packDepthToRGBA( v ).yx;
}
float unpackRGToDepth( const in highp vec2 v ) {
	return unpackRGBAToDepth( vec4( v.xy, 0.0, 0.0 ) );
}
vec4 pack2HalfToRGBA( vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,ru=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,su=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,au=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,ou=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,Au=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,lu=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,cu=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		return step( compare, unpackRGBAToDepth( texture2D( depths, uv ) ) );
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow (sampler2D shadow, vec2 uv, float compare ){
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		float hard_shadow = step( compare , distribution.x );
		if (hard_shadow != 1.0 ) {
			float distance = compare - distribution.x ;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return shadow;
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
		vec3 lightToPosition = shadowCoord.xyz;
		float dp = ( length( lightToPosition ) - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );		dp += shadowBias;
		vec3 bd3D = normalize( lightToPosition );
		#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
			vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
			return (
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
			) * ( 1.0 / 9.0 );
		#else
			return texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
		#endif
	}
#endif`,uu=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,du=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,fu=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,hu=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,pu=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,gu=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,mu=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,Eu=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,Su=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,vu=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,xu=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 OptimizedCineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color *= toneMappingExposure;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	return color;
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,Mu=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,Iu=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
		vec3 refractedRayExit = position + transmissionRay;
		vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
		vec2 refractionCoords = ndcPos.xy / ndcPos.w;
		refractionCoords += 1.0;
		refractionCoords /= 2.0;
		vec4 transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
		vec3 transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,yu=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Cu=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Tu=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,Pu=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,bu=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,Bu=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,ku=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Ru=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,_u=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Lu=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,wu=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Ju=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( 1.0 );
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	float fragCoordZ = 0.5 * vHighPrecisionZW[0] / vHighPrecisionZW[1] + 0.5;
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#endif
}`,Ou=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Du=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( 1.0 );
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,Hu=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Qu=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Nu=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Gu=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,zu=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,ju=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Uu=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Xu=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Zu=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,Fu=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Wu=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,Yu=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), opacity );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,Vu=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,qu=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Ku=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,$u=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,ed=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,td=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec4 diffuseColor = vec4( diffuse, opacity );
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,nd=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,id=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,rd=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,sd=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,ad=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix * vec4( 0.0, 0.0, 0.0, 1.0 );
	vec2 scale;
	scale.x = length( vec3( modelMatrix[ 0 ].x, modelMatrix[ 0 ].y, modelMatrix[ 0 ].z ) );
	scale.y = length( vec3( modelMatrix[ 1 ].x, modelMatrix[ 1 ].y, modelMatrix[ 1 ].z ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,od=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,Pe={alphahash_fragment:Bl,alphahash_pars_fragment:kl,alphamap_fragment:Rl,alphamap_pars_fragment:_l,alphatest_fragment:Ll,alphatest_pars_fragment:wl,aomap_fragment:Jl,aomap_pars_fragment:Ol,batching_pars_vertex:Dl,batching_vertex:Hl,begin_vertex:Ql,beginnormal_vertex:Nl,bsdfs:Gl,iridescence_fragment:zl,bumpmap_pars_fragment:jl,clipping_planes_fragment:Ul,clipping_planes_pars_fragment:Xl,clipping_planes_pars_vertex:Zl,clipping_planes_vertex:Fl,color_fragment:Wl,color_pars_fragment:Yl,color_pars_vertex:Vl,color_vertex:ql,common:Kl,cube_uv_reflection_fragment:$l,defaultnormal_vertex:ec,displacementmap_pars_vertex:tc,displacementmap_vertex:nc,emissivemap_fragment:ic,emissivemap_pars_fragment:rc,colorspace_fragment:sc,colorspace_pars_fragment:ac,envmap_fragment:oc,envmap_common_pars_fragment:Ac,envmap_pars_fragment:lc,envmap_pars_vertex:cc,envmap_physical_pars_fragment:Mc,envmap_vertex:uc,fog_vertex:dc,fog_pars_vertex:fc,fog_fragment:hc,fog_pars_fragment:pc,gradientmap_pars_fragment:gc,lightmap_fragment:mc,lightmap_pars_fragment:Ec,lights_lambert_fragment:Sc,lights_lambert_pars_fragment:vc,lights_pars_begin:xc,lights_toon_fragment:Ic,lights_toon_pars_fragment:yc,lights_phong_fragment:Cc,lights_phong_pars_fragment:Tc,lights_physical_fragment:Pc,lights_physical_pars_fragment:bc,lights_fragment_begin:Bc,lights_fragment_maps:kc,lights_fragment_end:Rc,logdepthbuf_fragment:_c,logdepthbuf_pars_fragment:Lc,logdepthbuf_pars_vertex:wc,logdepthbuf_vertex:Jc,map_fragment:Oc,map_pars_fragment:Dc,map_particle_fragment:Hc,map_particle_pars_fragment:Qc,metalnessmap_fragment:Nc,metalnessmap_pars_fragment:Gc,morphcolor_vertex:zc,morphnormal_vertex:jc,morphtarget_pars_vertex:Uc,morphtarget_vertex:Xc,normal_fragment_begin:Zc,normal_fragment_maps:Fc,normal_pars_fragment:Wc,normal_pars_vertex:Yc,normal_vertex:Vc,normalmap_pars_fragment:qc,clearcoat_normal_fragment_begin:Kc,clearcoat_normal_fragment_maps:$c,clearcoat_pars_fragment:eu,iridescence_pars_fragment:tu,opaque_fragment:nu,packing:iu,premultiplied_alpha_fragment:ru,project_vertex:su,dithering_fragment:au,dithering_pars_fragment:ou,roughnessmap_fragment:Au,roughnessmap_pars_fragment:lu,shadowmap_pars_fragment:cu,shadowmap_pars_vertex:uu,shadowmap_vertex:du,shadowmask_pars_fragment:fu,skinbase_vertex:hu,skinning_pars_vertex:pu,skinning_vertex:gu,skinnormal_vertex:mu,specularmap_fragment:Eu,specularmap_pars_fragment:Su,tonemapping_fragment:vu,tonemapping_pars_fragment:xu,transmission_fragment:Mu,transmission_pars_fragment:Iu,uv_pars_fragment:yu,uv_pars_vertex:Cu,uv_vertex:Tu,worldpos_vertex:Pu,background_vert:bu,background_frag:Bu,backgroundCube_vert:ku,backgroundCube_frag:Ru,cube_vert:_u,cube_frag:Lu,depth_vert:wu,depth_frag:Ju,distanceRGBA_vert:Ou,distanceRGBA_frag:Du,equirect_vert:Hu,equirect_frag:Qu,linedashed_vert:Nu,linedashed_frag:Gu,meshbasic_vert:zu,meshbasic_frag:ju,meshlambert_vert:Uu,meshlambert_frag:Xu,meshmatcap_vert:Zu,meshmatcap_frag:Fu,meshnormal_vert:Wu,meshnormal_frag:Yu,meshphong_vert:Vu,meshphong_frag:qu,meshphysical_vert:Ku,meshphysical_frag:$u,meshtoon_vert:ed,meshtoon_frag:td,points_vert:nd,points_frag:id,shadow_vert:rd,shadow_frag:sd,sprite_vert:ad,sprite_frag:od},te={common:{diffuse:{value:new Ne(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Oe},alphaMap:{value:null},alphaMapTransform:{value:new Oe},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Oe}},envmap:{envMap:{value:null},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Oe}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Oe}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Oe},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Oe},normalScale:{value:new Fe(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Oe},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Oe}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Oe}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Oe}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Ne(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new Ne(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Oe},alphaTest:{value:0},uvTransform:{value:new Oe}},sprite:{diffuse:{value:new Ne(16777215)},opacity:{value:1},center:{value:new Fe(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Oe},alphaMap:{value:null},alphaMapTransform:{value:new Oe},alphaTest:{value:0}}},Dt={basic:{uniforms:Et([te.common,te.specularmap,te.envmap,te.aomap,te.lightmap,te.fog]),vertexShader:Pe.meshbasic_vert,fragmentShader:Pe.meshbasic_frag},lambert:{uniforms:Et([te.common,te.specularmap,te.envmap,te.aomap,te.lightmap,te.emissivemap,te.bumpmap,te.normalmap,te.displacementmap,te.fog,te.lights,{emissive:{value:new Ne(0)}}]),vertexShader:Pe.meshlambert_vert,fragmentShader:Pe.meshlambert_frag},phong:{uniforms:Et([te.common,te.specularmap,te.envmap,te.aomap,te.lightmap,te.emissivemap,te.bumpmap,te.normalmap,te.displacementmap,te.fog,te.lights,{emissive:{value:new Ne(0)},specular:{value:new Ne(1118481)},shininess:{value:30}}]),vertexShader:Pe.meshphong_vert,fragmentShader:Pe.meshphong_frag},standard:{uniforms:Et([te.common,te.envmap,te.aomap,te.lightmap,te.emissivemap,te.bumpmap,te.normalmap,te.displacementmap,te.roughnessmap,te.metalnessmap,te.fog,te.lights,{emissive:{value:new Ne(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:Pe.meshphysical_vert,fragmentShader:Pe.meshphysical_frag},toon:{uniforms:Et([te.common,te.aomap,te.lightmap,te.emissivemap,te.bumpmap,te.normalmap,te.displacementmap,te.gradientmap,te.fog,te.lights,{emissive:{value:new Ne(0)}}]),vertexShader:Pe.meshtoon_vert,fragmentShader:Pe.meshtoon_frag},matcap:{uniforms:Et([te.common,te.bumpmap,te.normalmap,te.displacementmap,te.fog,{matcap:{value:null}}]),vertexShader:Pe.meshmatcap_vert,fragmentShader:Pe.meshmatcap_frag},points:{uniforms:Et([te.points,te.fog]),vertexShader:Pe.points_vert,fragmentShader:Pe.points_frag},dashed:{uniforms:Et([te.common,te.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:Pe.linedashed_vert,fragmentShader:Pe.linedashed_frag},depth:{uniforms:Et([te.common,te.displacementmap]),vertexShader:Pe.depth_vert,fragmentShader:Pe.depth_frag},normal:{uniforms:Et([te.common,te.bumpmap,te.normalmap,te.displacementmap,{opacity:{value:1}}]),vertexShader:Pe.meshnormal_vert,fragmentShader:Pe.meshnormal_frag},sprite:{uniforms:Et([te.sprite,te.fog]),vertexShader:Pe.sprite_vert,fragmentShader:Pe.sprite_frag},background:{uniforms:{uvTransform:{value:new Oe},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:Pe.background_vert,fragmentShader:Pe.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1}},vertexShader:Pe.backgroundCube_vert,fragmentShader:Pe.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:Pe.cube_vert,fragmentShader:Pe.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:Pe.equirect_vert,fragmentShader:Pe.equirect_frag},distanceRGBA:{uniforms:Et([te.common,te.displacementmap,{referencePosition:{value:new J},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:Pe.distanceRGBA_vert,fragmentShader:Pe.distanceRGBA_frag},shadow:{uniforms:Et([te.lights,te.fog,{color:{value:new Ne(0)},opacity:{value:1}}]),vertexShader:Pe.shadow_vert,fragmentShader:Pe.shadow_frag}};Dt.physical={uniforms:Et([Dt.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Oe},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Oe},clearcoatNormalScale:{value:new Fe(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Oe},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Oe},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Oe},sheen:{value:0},sheenColor:{value:new Ne(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Oe},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Oe},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Oe},transmissionSamplerSize:{value:new Fe},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Oe},attenuationDistance:{value:0},attenuationColor:{value:new Ne(0)},specularColor:{value:new Ne(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Oe},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Oe},anisotropyVector:{value:new Fe},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Oe}}]),vertexShader:Pe.meshphysical_vert,fragmentShader:Pe.meshphysical_frag};var Ki={r:0,b:0,g:0};function Ad(t,e,n,i,r,s,o){const a=new Ne(0);let A=s===!0?0:1,l,c,d=null,f=0,p=null;function E(h,u){let v=!1,y=u.isScene===!0?u.background:null;y&&y.isTexture&&(y=(u.backgroundBlurriness>0?n:e).get(y)),y===null?g(a,A):y&&y.isColor&&(g(y,1),v=!0);const T=t.xr.getEnvironmentBlendMode();T==="additive"?i.buffers.color.setClear(0,0,0,1,o):T==="alpha-blend"&&i.buffers.color.setClear(0,0,0,0,o),(t.autoClear||v)&&t.clear(t.autoClearColor,t.autoClearDepth,t.autoClearStencil),y&&(y.isCubeTexture||y.mapping===vi)?(c===void 0&&(c=new Yt(new Fr(1,1,1),new Vt({name:"BackgroundCubeMaterial",uniforms:Yn(Dt.backgroundCube.uniforms),vertexShader:Dt.backgroundCube.vertexShader,fragmentShader:Dt.backgroundCube.fragmentShader,side:ut,depthTest:!1,depthWrite:!1,fog:!1})),c.geometry.deleteAttribute("normal"),c.geometry.deleteAttribute("uv"),c.onBeforeRender=function(_,C,B){this.matrixWorld.copyPosition(B.matrixWorld)},Object.defineProperty(c.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),r.update(c)),c.material.uniforms.envMap.value=y,c.material.uniforms.flipEnvMap.value=y.isCubeTexture&&y.isRenderTargetTexture===!1?-1:1,c.material.uniforms.backgroundBlurriness.value=u.backgroundBlurriness,c.material.uniforms.backgroundIntensity.value=u.backgroundIntensity,c.material.toneMapped=je.getTransfer(y.colorSpace)!==We,(d!==y||f!==y.version||p!==t.toneMapping)&&(c.material.needsUpdate=!0,d=y,f=y.version,p=t.toneMapping),c.layers.enableAll(),h.unshift(c,c.geometry,c.material,0,0,null)):y&&y.isTexture&&(l===void 0&&(l=new Yt(new bl(2,2),new Vt({name:"BackgroundMaterial",uniforms:Yn(Dt.background.uniforms),vertexShader:Dt.background.vertexShader,fragmentShader:Dt.background.fragmentShader,side:$t,depthTest:!1,depthWrite:!1,fog:!1})),l.geometry.deleteAttribute("normal"),Object.defineProperty(l.material,"map",{get:function(){return this.uniforms.t2D.value}}),r.update(l)),l.material.uniforms.t2D.value=y,l.material.uniforms.backgroundIntensity.value=u.backgroundIntensity,l.material.toneMapped=je.getTransfer(y.colorSpace)!==We,y.matrixAutoUpdate===!0&&y.updateMatrix(),l.material.uniforms.uvTransform.value.copy(y.matrix),(d!==y||f!==y.version||p!==t.toneMapping)&&(l.material.needsUpdate=!0,d=y,f=y.version,p=t.toneMapping),l.layers.enableAll(),h.unshift(l,l.geometry,l.material,0,0,null))}function g(h,u){h.getRGB(Ki,Da(t)),i.buffers.color.setClear(Ki.r,Ki.g,Ki.b,u,o)}return{getClearColor:function(){return a},setClearColor:function(h,u=1){a.set(h),A=u,g(a,A)},getClearAlpha:function(){return A},setClearAlpha:function(h){A=h,g(a,A)},render:E}}function ld(t,e,n,i){const r=t.getParameter(t.MAX_VERTEX_ATTRIBS),s=i.isWebGL2?null:e.get("OES_vertex_array_object"),o=i.isWebGL2||s!==null,a={},A=h(null);let l=A,c=!1;function d(b,Q,G,q,U){let j=!1;if(o){const X=g(q,G,Q);l!==X&&(l=X,p(l.object)),j=u(b,q,G,U),j&&v(b,q,G,U)}else{const X=Q.wireframe===!0;(l.geometry!==q.id||l.program!==G.id||l.wireframe!==X)&&(l.geometry=q.id,l.program=G.id,l.wireframe=X,j=!0)}U!==null&&n.update(U,t.ELEMENT_ARRAY_BUFFER),(j||c)&&(c=!1,z(b,Q,G,q),U!==null&&t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,n.get(U).buffer))}function f(){return i.isWebGL2?t.createVertexArray():s.createVertexArrayOES()}function p(b){return i.isWebGL2?t.bindVertexArray(b):s.bindVertexArrayOES(b)}function E(b){return i.isWebGL2?t.deleteVertexArray(b):s.deleteVertexArrayOES(b)}function g(b,Q,G){const q=G.wireframe===!0;let U=a[b.id];U===void 0&&(U={},a[b.id]=U);let j=U[Q.id];j===void 0&&(j={},U[Q.id]=j);let X=j[q];return X===void 0&&(X=h(f()),j[q]=X),X}function h(b){const Q=[],G=[],q=[];for(let U=0;U<r;U++)Q[U]=0,G[U]=0,q[U]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:Q,enabledAttributes:G,attributeDivisors:q,object:b,attributes:{},index:null}}function u(b,Q,G,q){const U=l.attributes,j=Q.attributes;let X=0;const ee=G.getAttributes();for(const k in ee)if(ee[k].location>=0){const re=U[k];let se=j[k];if(se===void 0&&(k==="instanceMatrix"&&b.instanceMatrix&&(se=b.instanceMatrix),k==="instanceColor"&&b.instanceColor&&(se=b.instanceColor)),re===void 0||re.attribute!==se||se&&re.data!==se.data)return!0;X++}return l.attributesNum!==X||l.index!==q}function v(b,Q,G,q){const U={},j=Q.attributes;let X=0;const ee=G.getAttributes();for(const k in ee)if(ee[k].location>=0){let re=j[k];re===void 0&&(k==="instanceMatrix"&&b.instanceMatrix&&(re=b.instanceMatrix),k==="instanceColor"&&b.instanceColor&&(re=b.instanceColor));const se={};se.attribute=re,re&&re.data&&(se.data=re.data),U[k]=se,X++}l.attributes=U,l.attributesNum=X,l.index=q}function y(){const b=l.newAttributes;for(let Q=0,G=b.length;Q<G;Q++)b[Q]=0}function T(b){_(b,0)}function _(b,Q){const G=l.newAttributes,q=l.enabledAttributes,U=l.attributeDivisors;G[b]=1,q[b]===0&&(t.enableVertexAttribArray(b),q[b]=1),U[b]!==Q&&((i.isWebGL2?t:e.get("ANGLE_instanced_arrays"))[i.isWebGL2?"vertexAttribDivisor":"vertexAttribDivisorANGLE"](b,Q),U[b]=Q)}function C(){const b=l.newAttributes,Q=l.enabledAttributes;for(let G=0,q=Q.length;G<q;G++)Q[G]!==b[G]&&(t.disableVertexAttribArray(G),Q[G]=0)}function B(b,Q,G,q,U,j,X){X===!0?t.vertexAttribIPointer(b,Q,G,U,j):t.vertexAttribPointer(b,Q,G,q,U,j)}function z(b,Q,G,q){if(i.isWebGL2===!1&&(b.isInstancedMesh||q.isInstancedBufferGeometry)&&e.get("ANGLE_instanced_arrays")===null)return;y();const U=q.attributes,j=G.getAttributes(),X=Q.defaultAttributeValues;for(const ee in j){const k=j[ee];if(k.location>=0){let Z=U[ee];if(Z===void 0&&(ee==="instanceMatrix"&&b.instanceMatrix&&(Z=b.instanceMatrix),ee==="instanceColor"&&b.instanceColor&&(Z=b.instanceColor)),Z!==void 0){const re=Z.normalized,se=Z.itemSize,me=n.get(Z);if(me===void 0)continue;const xe=me.buffer,Le=me.type,Me=me.bytesPerElement,ze=i.isWebGL2===!0&&(Le===t.INT||Le===t.UNSIGNED_INT||Z.gpuType===Is);if(Z.isInterleavedBufferAttribute){const L=Z.data,St=L.stride,He=Z.offset;if(L.isInstancedInterleavedBuffer){for(let he=0;he<k.locationSize;he++)_(k.location+he,L.meshPerAttribute);b.isInstancedMesh!==!0&&q._maxInstanceCount===void 0&&(q._maxInstanceCount=L.meshPerAttribute*L.count)}else for(let he=0;he<k.locationSize;he++)T(k.location+he);t.bindBuffer(t.ARRAY_BUFFER,xe);for(let he=0;he<k.locationSize;he++)B(k.location+he,se/k.locationSize,Le,re,St*Me,(He+se/k.locationSize*he)*Me,ze)}else{if(Z.isInstancedBufferAttribute){for(let L=0;L<k.locationSize;L++)_(k.location+L,Z.meshPerAttribute);b.isInstancedMesh!==!0&&q._maxInstanceCount===void 0&&(q._maxInstanceCount=Z.meshPerAttribute*Z.count)}else for(let L=0;L<k.locationSize;L++)T(k.location+L);t.bindBuffer(t.ARRAY_BUFFER,xe);for(let L=0;L<k.locationSize;L++)B(k.location+L,se/k.locationSize,Le,re,se*Me,se/k.locationSize*L*Me,ze)}}else if(X!==void 0){const re=X[ee];if(re!==void 0)switch(re.length){case 2:t.vertexAttrib2fv(k.location,re);break;case 3:t.vertexAttrib3fv(k.location,re);break;case 4:t.vertexAttrib4fv(k.location,re);break;default:t.vertexAttrib1fv(k.location,re)}}}}C()}function M(){W();for(const b in a){const Q=a[b];for(const G in Q){const q=Q[G];for(const U in q)E(q[U].object),delete q[U];delete Q[G]}delete a[b]}}function I(b){if(a[b.id]===void 0)return;const Q=a[b.id];for(const G in Q){const q=Q[G];for(const U in q)E(q[U].object),delete q[U];delete Q[G]}delete a[b.id]}function H(b){for(const Q in a){const G=a[Q];if(G[b.id]===void 0)continue;const q=G[b.id];for(const U in q)E(q[U].object),delete q[U];delete G[b.id]}}function W(){Y(),c=!0,l!==A&&(l=A,p(l.object))}function Y(){A.geometry=null,A.program=null,A.wireframe=!1}return{setup:d,reset:W,resetDefaultState:Y,dispose:M,releaseStatesOfGeometry:I,releaseStatesOfProgram:H,initAttributes:y,enableAttribute:T,disableUnusedAttributes:C}}function cd(t,e,n,i){const r=i.isWebGL2;let s;function o(c){s=c}function a(c,d){t.drawArrays(s,c,d),n.update(d,s,1)}function A(c,d,f){if(f===0)return;let p,E;if(r)p=t,E="drawArraysInstanced";else if(p=e.get("ANGLE_instanced_arrays"),E="drawArraysInstancedANGLE",p===null){console.error("THREE.WebGLBufferRenderer: using THREE.InstancedBufferGeometry but hardware does not support extension ANGLE_instanced_arrays.");return}p[E](s,c,d,f),n.update(d,s,f)}function l(c,d,f){if(f===0)return;const p=e.get("WEBGL_multi_draw");if(p===null)for(let E=0;E<f;E++)this.render(c[E],d[E]);else{p.multiDrawArraysWEBGL(s,c,0,d,0,f);let E=0;for(let g=0;g<f;g++)E+=d[g];n.update(E,s,1)}}this.setMode=o,this.render=a,this.renderInstances=A,this.renderMultiDraw=l}function ud(t,e,n){let i;function r(){if(i!==void 0)return i;if(e.has("EXT_texture_filter_anisotropic")===!0){const B=e.get("EXT_texture_filter_anisotropic");i=t.getParameter(B.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else i=0;return i}function s(B){if(B==="highp"){if(t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.HIGH_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.HIGH_FLOAT).precision>0)return"highp";B="mediump"}return B==="mediump"&&t.getShaderPrecisionFormat(t.VERTEX_SHADER,t.MEDIUM_FLOAT).precision>0&&t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}const o=typeof WebGL2RenderingContext<"u"&&t.constructor.name==="WebGL2RenderingContext";let a=n.precision!==void 0?n.precision:"highp";const A=s(a);A!==a&&(console.warn("THREE.WebGLRenderer:",a,"not supported, using",A,"instead."),a=A);const l=o||e.has("WEBGL_draw_buffers"),c=n.logarithmicDepthBuffer===!0,d=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS),f=t.getParameter(t.MAX_VERTEX_TEXTURE_IMAGE_UNITS),p=t.getParameter(t.MAX_TEXTURE_SIZE),E=t.getParameter(t.MAX_CUBE_MAP_TEXTURE_SIZE),g=t.getParameter(t.MAX_VERTEX_ATTRIBS),h=t.getParameter(t.MAX_VERTEX_UNIFORM_VECTORS),u=t.getParameter(t.MAX_VARYING_VECTORS),v=t.getParameter(t.MAX_FRAGMENT_UNIFORM_VECTORS),y=f>0,T=o||e.has("OES_texture_float"),_=y&&T,C=o?t.getParameter(t.MAX_SAMPLES):0;return{isWebGL2:o,drawBuffers:l,getMaxAnisotropy:r,getMaxPrecision:s,precision:a,logarithmicDepthBuffer:c,maxTextures:d,maxVertexTextures:f,maxTextureSize:p,maxCubemapSize:E,maxAttributes:g,maxVertexUniforms:h,maxVaryings:u,maxFragmentUniforms:v,vertexTextures:y,floatFragmentTextures:T,floatVertexTextures:_,maxSamples:C}}function dd(t){const e=this;let n=null,i=0,r=!1,s=!1;const o=new xn,a=new Oe,A={value:null,needsUpdate:!1};this.uniform=A,this.numPlanes=0,this.numIntersection=0,this.init=function(d,f){const p=d.length!==0||f||i!==0||r;return r=f,i=d.length,p},this.beginShadows=function(){s=!0,c(null)},this.endShadows=function(){s=!1},this.setGlobalState=function(d,f){n=c(d,f,0)},this.setState=function(d,f,p){const E=d.clippingPlanes,g=d.clipIntersection,h=d.clipShadows,u=t.get(d);if(!r||E===null||E.length===0||s&&!h)s?c(null):l();else{const v=s?0:i,y=v*4;let T=u.clippingState||null;A.value=T,T=c(E,f,y,p);for(let _=0;_!==y;++_)T[_]=n[_];u.clippingState=T,this.numIntersection=g?this.numPlanes:0,this.numPlanes+=v}};function l(){A.value!==n&&(A.value=n,A.needsUpdate=i>0),e.numPlanes=i,e.numIntersection=0}function c(d,f,p,E){const g=d!==null?d.length:0;let h=null;if(g!==0){if(h=A.value,E!==!0||h===null){const u=p+g*4,v=f.matrixWorldInverse;a.getNormalMatrix(v),(h===null||h.length<u)&&(h=new Float32Array(u));for(let y=0,T=p;y!==g;++y,T+=4)o.copy(d[y]).applyMatrix4(v,a),o.normal.toArray(h,T),h[T+3]=o.constant}A.value=h,A.needsUpdate=!0}return e.numPlanes=g,e.numIntersection=0,h}}function fd(t){let e=new WeakMap;function n(o,a){return a===hr?o.mapping=bn:a===pr&&(o.mapping=Bn),o}function i(o){if(o&&o.isTexture){const a=o.mapping;if(a===hr||a===pr)if(e.has(o)){const A=e.get(o).texture;return n(A,o.mapping)}else{const A=o.image;if(A&&A.height>0){const l=new yl(A.height/2);return l.fromEquirectangularTexture(t,o),e.set(o,l),o.addEventListener("dispose",r),n(l.texture,o.mapping)}else return null}}return o}function r(o){const a=o.target;a.removeEventListener("dispose",r);const A=e.get(a);A!==void 0&&(e.delete(a),A.dispose())}function s(){e=new WeakMap}return{get:i,dispose:s}}var hd=class extends Ha{constructor(t=-1,e=1,n=1,i=-1,r=.1,s=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=t,this.right=e,this.top=n,this.bottom=i,this.near=r,this.far=s,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.left=t.left,this.right=t.right,this.top=t.top,this.bottom=t.bottom,this.near=t.near,this.far=t.far,this.zoom=t.zoom,this.view=t.view===null?null:Object.assign({},t.view),this}setViewOffset(t,e,n,i,r,s){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=i,this.view.width=r,this.view.height=s,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=(this.right-this.left)/(2*this.zoom),e=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,i=(this.top+this.bottom)/2;let r=n-t,s=n+t,o=i+e,a=i-e;if(this.view!==null&&this.view.enabled){const A=(this.right-this.left)/this.view.fullWidth/this.zoom,l=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=A*this.view.offsetX,s=r+A*this.view.width,o-=l*this.view.offsetY,a=o-l*this.view.height}this.projectionMatrix.makeOrthographic(r,s,o,a,this.near,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.zoom=this.zoom,e.object.left=this.left,e.object.right=this.right,e.object.top=this.top,e.object.bottom=this.bottom,e.object.near=this.near,e.object.far=this.far,this.view!==null&&(e.object.view=Object.assign({},this.view)),e}},Kn=4,za=[.125,.215,.35,.446,.526,.582],In=20,Yr=new hd,ja=new Ne,Vr=null,qr=0,Kr=0,yn=(1+Math.sqrt(5))/2,$n=1/yn,Ua=[new J(1,1,1),new J(-1,1,1),new J(1,1,-1),new J(-1,1,-1),new J(0,yn,$n),new J(0,yn,-$n),new J($n,0,yn),new J(-$n,0,yn),new J(yn,$n,0),new J(-yn,$n,0)],Xa=class{constructor(t){this._renderer=t,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._lodPlanes=[],this._sizeLods=[],this._sigmas=[],this._blurMaterial=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._compileMaterial(this._blurMaterial)}fromScene(t,e=0,n=.1,i=100){Vr=this._renderer.getRenderTarget(),qr=this._renderer.getActiveCubeFace(),Kr=this._renderer.getActiveMipmapLevel(),this._setSize(256);const r=this._allocateTargets();return r.depthBuffer=!0,this._sceneToCubeUV(t,n,i,r),e>0&&this._blur(r,0,0,e),this._applyPMREM(r),this._cleanup(r),r}fromEquirectangular(t,e=null){return this._fromTexture(t,e)}fromCubemap(t,e=null){return this._fromTexture(t,e)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Wa(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=Fa(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose()}_setSize(t){this._lodMax=Math.floor(Math.log2(t)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let t=0;t<this._lodPlanes.length;t++)this._lodPlanes[t].dispose()}_cleanup(t){this._renderer.setRenderTarget(Vr,qr,Kr),t.scissorTest=!1,$i(t,0,0,t.width,t.height)}_fromTexture(t,e){t.mapping===bn||t.mapping===Bn?this._setSize(t.image.length===0?16:t.image[0].width||t.image[0].image.width):this._setSize(t.image.width/4),Vr=this._renderer.getRenderTarget(),qr=this._renderer.getActiveCubeFace(),Kr=this._renderer.getActiveMipmapLevel();const n=e||this._allocateTargets();return this._textureToCubeUV(t,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const t=3*Math.max(this._cubeSize,112),e=4*this._cubeSize,n={magFilter:yt,minFilter:yt,generateMipmaps:!1,type:ri,format:kt,colorSpace:zt,depthBuffer:!1},i=Za(t,e,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==t||this._pingPongRenderTarget.height!==e){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=Za(t,e,n);const{_lodMax:r}=this;({sizeLods:this._sizeLods,lodPlanes:this._lodPlanes,sigmas:this._sigmas}=pd(r)),this._blurMaterial=gd(r,t,e)}return i}_compileMaterial(t){const e=new Yt(this._lodPlanes[0],t);this._renderer.compile(e,Yr)}_sceneToCubeUV(t,e,n,i){const o=new Pt(90,1,e,n),a=[1,-1,1,1,1,1],A=[1,1,1,-1,-1,-1],l=this._renderer,c=l.autoClear,d=l.toneMapping;l.getClearColor(ja),l.toneMapping=tn,l.autoClear=!1;const f=new Ba({name:"PMREM.Background",side:ut,depthWrite:!1,depthTest:!1}),p=new Yt(new Fr,f);let E=!1;const g=t.background;g?g.isColor&&(f.color.copy(g),t.background=null,E=!0):(f.color.copy(ja),E=!0);for(let h=0;h<6;h++){const u=h%3;u===0?(o.up.set(0,a[h],0),o.lookAt(A[h],0,0)):u===1?(o.up.set(0,0,a[h]),o.lookAt(0,A[h],0)):(o.up.set(0,a[h],0),o.lookAt(0,0,A[h]));const v=this._cubeSize;$i(i,u*v,h>2?v:0,v,v),l.setRenderTarget(i),E&&l.render(p,o),l.render(t,o)}p.geometry.dispose(),p.material.dispose(),l.toneMapping=d,l.autoClear=c,t.background=g}_textureToCubeUV(t,e){const n=this._renderer,i=t.mapping===bn||t.mapping===Bn;i?(this._cubemapMaterial===null&&(this._cubemapMaterial=Wa()),this._cubemapMaterial.uniforms.flipEnvMap.value=t.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=Fa());const r=i?this._cubemapMaterial:this._equirectMaterial,s=new Yt(this._lodPlanes[0],r),o=r.uniforms;o.envMap.value=t;const a=this._cubeSize;$i(e,0,0,3*a,2*a),n.setRenderTarget(e),n.render(s,Yr)}_applyPMREM(t){const e=this._renderer,n=e.autoClear;e.autoClear=!1;for(let i=1;i<this._lodPlanes.length;i++){const r=Math.sqrt(this._sigmas[i]*this._sigmas[i]-this._sigmas[i-1]*this._sigmas[i-1]),s=Ua[(i-1)%Ua.length];this._blur(t,i-1,i,r,s)}e.autoClear=n}_blur(t,e,n,i,r){const s=this._pingPongRenderTarget;this._halfBlur(t,s,e,n,i,"latitudinal",r),this._halfBlur(s,t,n,n,i,"longitudinal",r)}_halfBlur(t,e,n,i,r,s,o){const a=this._renderer,A=this._blurMaterial;s!=="latitudinal"&&s!=="longitudinal"&&console.error("blur direction must be either latitudinal or longitudinal!");const l=3,c=new Yt(this._lodPlanes[i],A),d=A.uniforms,f=this._sizeLods[n]-1,p=isFinite(r)?Math.PI/(2*f):2*Math.PI/(2*In-1),E=r/p,g=isFinite(r)?1+Math.floor(l*E):In;g>In&&console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${g} samples when the maximum is set to ${In}`);const h=[];let u=0;for(let C=0;C<In;++C){const B=C/E,z=Math.exp(-B*B/2);h.push(z),C===0?u+=z:C<g&&(u+=2*z)}for(let C=0;C<h.length;C++)h[C]=h[C]/u;d.envMap.value=t.texture,d.samples.value=g,d.weights.value=h,d.latitudinal.value=s==="latitudinal",o&&(d.poleAxis.value=o);const{_lodMax:v}=this;d.dTheta.value=p,d.mipInt.value=v-n;const y=this._sizeLods[i],T=3*y*(i>v-Kn?i-v+Kn:0),_=4*(this._cubeSize-y);$i(e,T,_,3*y,2*y),a.setRenderTarget(e),a.render(c,Yr)}};function pd(t){const e=[],n=[],i=[];let r=t;const s=t-Kn+1+za.length;for(let o=0;o<s;o++){const a=Math.pow(2,r);n.push(a);let A=1/a;o>t-Kn?A=za[o-t+Kn-1]:o===0&&(A=0),i.push(A);const l=1/(a-2),c=-l,d=1+l,f=[c,c,d,c,d,d,c,c,d,d,c,d],p=6,E=6,g=3,h=2,u=1,v=new Float32Array(g*E*p),y=new Float32Array(h*E*p),T=new Float32Array(u*E*p);for(let C=0;C<p;C++){const B=C%3*2/3-1,z=C>2?0:-1,M=[B,z,0,B+2/3,z,0,B+2/3,z+1,0,B,z,0,B+2/3,z+1,0,B,z+1,0];v.set(M,g*E*C),y.set(f,h*E*C);const I=[C,C,C,C,C,C];T.set(I,u*E*C)}const _=new un;_.setAttribute("position",new Ot(v,g)),_.setAttribute("uv",new Ot(y,h)),_.setAttribute("faceIndex",new Ot(T,u)),e.push(_),r>Kn&&r--}return{lodPlanes:e,sizeLods:n,sigmas:i}}function Za(t,e,n){const i=new mn(t,e,n);return i.texture.mapping=vi,i.texture.name="PMREM.cubeUv",i.scissorTest=!0,i}function $i(t,e,n,i,r){t.viewport.set(e,n,i,r),t.scissor.set(e,n,i,r)}function gd(t,e,n){const i=new Float32Array(In),r=new J(0,1,0);return new Vt({name:"SphericalGaussianBlur",defines:{n:In,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/n,CUBEUV_MAX_MIP:`${t}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:i},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:r}},vertexShader:$r(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:en,depthTest:!1,depthWrite:!1})}function Fa(){return new Vt({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:$r(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:en,depthTest:!1,depthWrite:!1})}function Wa(){return new Vt({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:$r(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:en,depthTest:!1,depthWrite:!1})}function $r(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function md(t){let e=new WeakMap,n=null;function i(a){if(a&&a.isTexture){const A=a.mapping,l=A===hr||A===pr,c=A===bn||A===Bn;if(l||c)if(a.isRenderTargetTexture&&a.needsPMREMUpdate===!0){a.needsPMREMUpdate=!1;let d=e.get(a);return n===null&&(n=new Xa(t)),d=l?n.fromEquirectangular(a,d):n.fromCubemap(a,d),e.set(a,d),d.texture}else{if(e.has(a))return e.get(a).texture;{const d=a.image;if(l&&d&&d.height>0||c&&d&&r(d)){n===null&&(n=new Xa(t));const f=l?n.fromEquirectangular(a):n.fromCubemap(a);return e.set(a,f),a.addEventListener("dispose",s),f.texture}else return null}}}return a}function r(a){let A=0;const l=6;for(let c=0;c<l;c++)a[c]!==void 0&&A++;return A===l}function s(a){const A=a.target;A.removeEventListener("dispose",s);const l=e.get(A);l!==void 0&&(e.delete(A),l.dispose())}function o(){e=new WeakMap,n!==null&&(n.dispose(),n=null)}return{get:i,dispose:o}}function Ed(t){const e={};function n(i){if(e[i]!==void 0)return e[i];let r;switch(i){case"WEBGL_depth_texture":r=t.getExtension("WEBGL_depth_texture")||t.getExtension("MOZ_WEBGL_depth_texture")||t.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":r=t.getExtension("EXT_texture_filter_anisotropic")||t.getExtension("MOZ_EXT_texture_filter_anisotropic")||t.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":r=t.getExtension("WEBGL_compressed_texture_s3tc")||t.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||t.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":r=t.getExtension("WEBGL_compressed_texture_pvrtc")||t.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:r=t.getExtension(i)}return e[i]=r,r}return{has:function(i){return n(i)!==null},init:function(i){i.isWebGL2?(n("EXT_color_buffer_float"),n("WEBGL_clip_cull_distance")):(n("WEBGL_depth_texture"),n("OES_texture_float"),n("OES_texture_half_float"),n("OES_texture_half_float_linear"),n("OES_standard_derivatives"),n("OES_element_index_uint"),n("OES_vertex_array_object"),n("ANGLE_instanced_arrays")),n("OES_texture_float_linear"),n("EXT_color_buffer_half_float"),n("WEBGL_multisampled_render_to_texture")},get:function(i){const r=n(i);return r===null&&console.warn("THREE.WebGLRenderer: "+i+" extension not supported."),r}}}function Sd(t,e,n,i){const r={},s=new WeakMap;function o(d){const f=d.target;f.index!==null&&e.remove(f.index);for(const E in f.attributes)e.remove(f.attributes[E]);for(const E in f.morphAttributes){const g=f.morphAttributes[E];for(let h=0,u=g.length;h<u;h++)e.remove(g[h])}f.removeEventListener("dispose",o),delete r[f.id];const p=s.get(f);p&&(e.remove(p),s.delete(f)),i.releaseStatesOfGeometry(f),f.isInstancedBufferGeometry===!0&&delete f._maxInstanceCount,n.memory.geometries--}function a(d,f){return r[f.id]===!0||(f.addEventListener("dispose",o),r[f.id]=!0,n.memory.geometries++),f}function A(d){const f=d.attributes;for(const E in f)e.update(f[E],t.ARRAY_BUFFER);const p=d.morphAttributes;for(const E in p){const g=p[E];for(let h=0,u=g.length;h<u;h++)e.update(g[h],t.ARRAY_BUFFER)}}function l(d){const f=[],p=d.index,E=d.attributes.position;let g=0;if(p!==null){const v=p.array;g=p.version;for(let y=0,T=v.length;y<T;y+=3){const _=v[y+0],C=v[y+1],B=v[y+2];f.push(_,C,C,B,B,_)}}else if(E!==void 0){const v=E.array;g=E.version;for(let y=0,T=v.length/3-1;y<T;y+=3){const _=y+0,C=y+1,B=y+2;f.push(_,C,C,B,B,_)}}else return;const h=new(ca(f)?Ra:ka)(f,1);h.version=g;const u=s.get(d);u&&e.remove(u),s.set(d,h)}function c(d){const f=s.get(d);if(f){const p=d.index;p!==null&&f.version<p.version&&l(d)}else l(d);return s.get(d)}return{get:a,update:A,getWireframeAttribute:c}}function vd(t,e,n,i){const r=i.isWebGL2;let s;function o(p){s=p}let a,A;function l(p){a=p.type,A=p.bytesPerElement}function c(p,E){t.drawElements(s,E,a,p*A),n.update(E,s,1)}function d(p,E,g){if(g===0)return;let h,u;if(r)h=t,u="drawElementsInstanced";else if(h=e.get("ANGLE_instanced_arrays"),u="drawElementsInstancedANGLE",h===null){console.error("THREE.WebGLIndexedBufferRenderer: using THREE.InstancedBufferGeometry but hardware does not support extension ANGLE_instanced_arrays.");return}h[u](s,E,a,p*A,g),n.update(E,s,g)}function f(p,E,g){if(g===0)return;const h=e.get("WEBGL_multi_draw");if(h===null)for(let u=0;u<g;u++)this.render(p[u]/A,E[u]);else{h.multiDrawElementsWEBGL(s,E,0,a,p,0,g);let u=0;for(let v=0;v<g;v++)u+=E[v];n.update(u,s,1)}}this.setMode=o,this.setIndex=l,this.render=c,this.renderInstances=d,this.renderMultiDraw=f}function xd(t){const e={geometries:0,textures:0},n={frame:0,calls:0,triangles:0,points:0,lines:0};function i(s,o,a){switch(n.calls++,o){case t.TRIANGLES:n.triangles+=a*(s/3);break;case t.LINES:n.lines+=a*(s/2);break;case t.LINE_STRIP:n.lines+=a*(s-1);break;case t.LINE_LOOP:n.lines+=a*s;break;case t.POINTS:n.points+=a*s;break;default:console.error("THREE.WebGLInfo: Unknown draw mode:",o);break}}function r(){n.calls=0,n.triangles=0,n.points=0,n.lines=0}return{memory:e,render:n,programs:null,autoReset:!0,reset:r,update:i}}function Md(t,e){return t[0]-e[0]}function Id(t,e){return Math.abs(e[1])-Math.abs(t[1])}function yd(t,e,n){const i={},r=new Float32Array(8),s=new WeakMap,o=new pt,a=[];for(let l=0;l<8;l++)a[l]=[l,0];function A(l,c,d){const f=l.morphTargetInfluences;if(e.isWebGL2===!0){const E=c.morphAttributes.position||c.morphAttributes.normal||c.morphAttributes.color,g=E!==void 0?E.length:0;let h=s.get(c);if(h===void 0||h.count!==g){let y=function(){b.dispose(),s.delete(c),c.removeEventListener("dispose",y)};var p=y;h!==void 0&&h.texture.dispose();const T=c.morphAttributes.position!==void 0,_=c.morphAttributes.normal!==void 0,C=c.morphAttributes.color!==void 0,B=c.morphAttributes.position||[],z=c.morphAttributes.normal||[],M=c.morphAttributes.color||[];let I=0;T===!0&&(I=1),_===!0&&(I=2),C===!0&&(I=3);let H=c.attributes.position.count*I,W=1;H>e.maxTextureSize&&(W=Math.ceil(H/e.maxTextureSize),H=e.maxTextureSize);const Y=new Float32Array(H*W*4*g),b=new ga(Y,H,W,g);b.type=sn,b.needsUpdate=!0;const Q=I*4;for(let G=0;G<g;G++){const q=B[G],U=z[G],j=M[G],X=H*W*4*G;for(let ee=0;ee<q.count;ee++){const k=ee*Q;T===!0&&(o.fromBufferAttribute(q,ee),Y[X+k+0]=o.x,Y[X+k+1]=o.y,Y[X+k+2]=o.z,Y[X+k+3]=0),_===!0&&(o.fromBufferAttribute(U,ee),Y[X+k+4]=o.x,Y[X+k+5]=o.y,Y[X+k+6]=o.z,Y[X+k+7]=0),C===!0&&(o.fromBufferAttribute(j,ee),Y[X+k+8]=o.x,Y[X+k+9]=o.y,Y[X+k+10]=o.z,Y[X+k+11]=j.itemSize===4?o.w:1)}}h={count:g,texture:b,size:new Fe(H,W)},s.set(c,h),c.addEventListener("dispose",y)}let u=0;for(let y=0;y<f.length;y++)u+=f[y];const v=c.morphTargetsRelative?1:1-u;d.getUniforms().setValue(t,"morphTargetBaseInfluence",v),d.getUniforms().setValue(t,"morphTargetInfluences",f),d.getUniforms().setValue(t,"morphTargetsTexture",h.texture,n),d.getUniforms().setValue(t,"morphTargetsTextureSize",h.size)}else{const E=f===void 0?0:f.length;let g=i[c.id];if(g===void 0||g.length!==E){g=[];for(let T=0;T<E;T++)g[T]=[T,0];i[c.id]=g}for(let T=0;T<E;T++){const _=g[T];_[0]=T,_[1]=f[T]}g.sort(Id);for(let T=0;T<8;T++)T<E&&g[T][1]?(a[T][0]=g[T][0],a[T][1]=g[T][1]):(a[T][0]=Number.MAX_SAFE_INTEGER,a[T][1]=0);a.sort(Md);const h=c.morphAttributes.position,u=c.morphAttributes.normal;let v=0;for(let T=0;T<8;T++){const _=a[T],C=_[0],B=_[1];C!==Number.MAX_SAFE_INTEGER&&B?(h&&c.getAttribute("morphTarget"+T)!==h[C]&&c.setAttribute("morphTarget"+T,h[C]),u&&c.getAttribute("morphNormal"+T)!==u[C]&&c.setAttribute("morphNormal"+T,u[C]),r[T]=B,v+=B):(h&&c.hasAttribute("morphTarget"+T)===!0&&c.deleteAttribute("morphTarget"+T),u&&c.hasAttribute("morphNormal"+T)===!0&&c.deleteAttribute("morphNormal"+T),r[T]=0)}const y=c.morphTargetsRelative?1:1-v;d.getUniforms().setValue(t,"morphTargetBaseInfluence",y),d.getUniforms().setValue(t,"morphTargetInfluences",r)}}return{update:A}}function Cd(t,e,n,i){let r=new WeakMap;function s(A){const l=i.render.frame,c=A.geometry,d=e.get(A,c);if(r.get(d)!==l&&(e.update(d),r.set(d,l)),A.isInstancedMesh&&(A.hasEventListener("dispose",a)===!1&&A.addEventListener("dispose",a),r.get(A)!==l&&(n.update(A.instanceMatrix,t.ARRAY_BUFFER),A.instanceColor!==null&&n.update(A.instanceColor,t.ARRAY_BUFFER),r.set(A,l))),A.isSkinnedMesh){const f=A.skeleton;r.get(f)!==l&&(f.update(),r.set(f,l))}return d}function o(){r=new WeakMap}function a(A){const l=A.target;l.removeEventListener("dispose",a),n.remove(l.instanceMatrix),l.instanceColor!==null&&n.remove(l.instanceColor)}return{update:s,dispose:o}}var Ya=class extends wt{constructor(t,e,n,i,r,s,o,a,A,l){if(l=l!==void 0?l:pn,l!==pn&&l!==kn)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");n===void 0&&l===pn&&(n=rn),n===void 0&&l===kn&&(n=hn),super(null,i,r,s,o,a,l,n,A),this.isDepthTexture=!0,this.image={width:t,height:e},this.magFilter=o!==void 0?o:dt,this.minFilter=a!==void 0?a:dt,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(t){return super.copy(t),this.compareFunction=t.compareFunction,this}toJSON(t){const e=super.toJSON(t);return this.compareFunction!==null&&(e.compareFunction=this.compareFunction),e}},Va=new wt,qa=new Ya(1,1);qa.compareFunction=aa;var Ka=new ga,$a=new ol,eo=new Qa,to=[],no=[],io=new Float32Array(16),ro=new Float32Array(9),so=new Float32Array(4);function ei(t,e,n){const i=t[0];if(i<=0||i>0)return t;const r=e*n;let s=to[r];if(s===void 0&&(s=new Float32Array(r),to[r]=s),e!==0){i.toArray(s,0);for(let o=1,a=0;o!==e;++o)a+=n,t[o].toArray(s,a)}return s}function nt(t,e){if(t.length!==e.length)return!1;for(let n=0,i=t.length;n<i;n++)if(t[n]!==e[n])return!1;return!0}function it(t,e){for(let n=0,i=e.length;n<i;n++)t[n]=e[n]}function er(t,e){let n=no[e];n===void 0&&(n=new Int32Array(e),no[e]=n);for(let i=0;i!==e;++i)n[i]=t.allocateTextureUnit();return n}function Td(t,e){const n=this.cache;n[0]!==e&&(t.uniform1f(this.addr,e),n[0]=e)}function Pd(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2f(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(nt(n,e))return;t.uniform2fv(this.addr,e),it(n,e)}}function bd(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3f(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else if(e.r!==void 0)(n[0]!==e.r||n[1]!==e.g||n[2]!==e.b)&&(t.uniform3f(this.addr,e.r,e.g,e.b),n[0]=e.r,n[1]=e.g,n[2]=e.b);else{if(nt(n,e))return;t.uniform3fv(this.addr,e),it(n,e)}}function Bd(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4f(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(nt(n,e))return;t.uniform4fv(this.addr,e),it(n,e)}}function kd(t,e){const n=this.cache,i=e.elements;if(i===void 0){if(nt(n,e))return;t.uniformMatrix2fv(this.addr,!1,e),it(n,e)}else{if(nt(n,i))return;so.set(i),t.uniformMatrix2fv(this.addr,!1,so),it(n,i)}}function Rd(t,e){const n=this.cache,i=e.elements;if(i===void 0){if(nt(n,e))return;t.uniformMatrix3fv(this.addr,!1,e),it(n,e)}else{if(nt(n,i))return;ro.set(i),t.uniformMatrix3fv(this.addr,!1,ro),it(n,i)}}function _d(t,e){const n=this.cache,i=e.elements;if(i===void 0){if(nt(n,e))return;t.uniformMatrix4fv(this.addr,!1,e),it(n,e)}else{if(nt(n,i))return;io.set(i),t.uniformMatrix4fv(this.addr,!1,io),it(n,i)}}function Ld(t,e){const n=this.cache;n[0]!==e&&(t.uniform1i(this.addr,e),n[0]=e)}function wd(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2i(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(nt(n,e))return;t.uniform2iv(this.addr,e),it(n,e)}}function Jd(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3i(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(nt(n,e))return;t.uniform3iv(this.addr,e),it(n,e)}}function Od(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4i(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(nt(n,e))return;t.uniform4iv(this.addr,e),it(n,e)}}function Dd(t,e){const n=this.cache;n[0]!==e&&(t.uniform1ui(this.addr,e),n[0]=e)}function Hd(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y)&&(t.uniform2ui(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(nt(n,e))return;t.uniform2uiv(this.addr,e),it(n,e)}}function Qd(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z)&&(t.uniform3ui(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else{if(nt(n,e))return;t.uniform3uiv(this.addr,e),it(n,e)}}function Nd(t,e){const n=this.cache;if(e.x!==void 0)(n[0]!==e.x||n[1]!==e.y||n[2]!==e.z||n[3]!==e.w)&&(t.uniform4ui(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(nt(n,e))return;t.uniform4uiv(this.addr,e),it(n,e)}}function Gd(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r);const s=this.type===t.SAMPLER_2D_SHADOW?qa:Va;n.setTexture2D(e||s,r)}function zd(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.setTexture3D(e||$a,r)}function jd(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.setTextureCube(e||eo,r)}function Ud(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.setTexture2DArray(e||Ka,r)}function Xd(t){switch(t){case 5126:return Td;case 35664:return Pd;case 35665:return bd;case 35666:return Bd;case 35674:return kd;case 35675:return Rd;case 35676:return _d;case 5124:case 35670:return Ld;case 35667:case 35671:return wd;case 35668:case 35672:return Jd;case 35669:case 35673:return Od;case 5125:return Dd;case 36294:return Hd;case 36295:return Qd;case 36296:return Nd;case 35678:case 36198:case 36298:case 36306:case 35682:return Gd;case 35679:case 36299:case 36307:return zd;case 35680:case 36300:case 36308:case 36293:return jd;case 36289:case 36303:case 36311:case 36292:return Ud}}function Zd(t,e){t.uniform1fv(this.addr,e)}function Fd(t,e){const n=ei(e,this.size,2);t.uniform2fv(this.addr,n)}function Wd(t,e){const n=ei(e,this.size,3);t.uniform3fv(this.addr,n)}function Yd(t,e){const n=ei(e,this.size,4);t.uniform4fv(this.addr,n)}function Vd(t,e){const n=ei(e,this.size,4);t.uniformMatrix2fv(this.addr,!1,n)}function qd(t,e){const n=ei(e,this.size,9);t.uniformMatrix3fv(this.addr,!1,n)}function Kd(t,e){const n=ei(e,this.size,16);t.uniformMatrix4fv(this.addr,!1,n)}function $d(t,e){t.uniform1iv(this.addr,e)}function ef(t,e){t.uniform2iv(this.addr,e)}function tf(t,e){t.uniform3iv(this.addr,e)}function nf(t,e){t.uniform4iv(this.addr,e)}function rf(t,e){t.uniform1uiv(this.addr,e)}function sf(t,e){t.uniform2uiv(this.addr,e)}function af(t,e){t.uniform3uiv(this.addr,e)}function of(t,e){t.uniform4uiv(this.addr,e)}function Af(t,e,n){const i=this.cache,r=e.length,s=er(n,r);nt(i,s)||(t.uniform1iv(this.addr,s),it(i,s));for(let o=0;o!==r;++o)n.setTexture2D(e[o]||Va,s[o])}function lf(t,e,n){const i=this.cache,r=e.length,s=er(n,r);nt(i,s)||(t.uniform1iv(this.addr,s),it(i,s));for(let o=0;o!==r;++o)n.setTexture3D(e[o]||$a,s[o])}function cf(t,e,n){const i=this.cache,r=e.length,s=er(n,r);nt(i,s)||(t.uniform1iv(this.addr,s),it(i,s));for(let o=0;o!==r;++o)n.setTextureCube(e[o]||eo,s[o])}function uf(t,e,n){const i=this.cache,r=e.length,s=er(n,r);nt(i,s)||(t.uniform1iv(this.addr,s),it(i,s));for(let o=0;o!==r;++o)n.setTexture2DArray(e[o]||Ka,s[o])}function df(t){switch(t){case 5126:return Zd;case 35664:return Fd;case 35665:return Wd;case 35666:return Yd;case 35674:return Vd;case 35675:return qd;case 35676:return Kd;case 5124:case 35670:return $d;case 35667:case 35671:return ef;case 35668:case 35672:return tf;case 35669:case 35673:return nf;case 5125:return rf;case 36294:return sf;case 36295:return af;case 36296:return of;case 35678:case 36198:case 36298:case 36306:case 35682:return Af;case 35679:case 36299:case 36307:return lf;case 35680:case 36300:case 36308:case 36293:return cf;case 36289:case 36303:case 36311:case 36292:return uf}}var ff=class{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.type=e.type,this.setValue=Xd(e.type)}},hf=class{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.type=e.type,this.size=e.size,this.setValue=df(e.type)}},pf=class{constructor(t){this.id=t,this.seq=[],this.map={}}setValue(t,e,n){const i=this.seq;for(let r=0,s=i.length;r!==s;++r){const o=i[r];o.setValue(t,e[o.id],n)}}},es=/(\w+)(\])?(\[|\.)?/g;function ao(t,e){t.seq.push(e),t.map[e.id]=e}function gf(t,e,n){const i=t.name,r=i.length;for(es.lastIndex=0;;){const s=es.exec(i),o=es.lastIndex;let a=s[1];const A=s[2]==="]",l=s[3];if(A&&(a=a|0),l===void 0||l==="["&&o+2===r){ao(n,l===void 0?new ff(a,t,e):new hf(a,t,e));break}else{let d=n.map[a];d===void 0&&(d=new pf(a),ao(n,d)),n=d}}}var tr=class{constructor(t,e){this.seq=[],this.map={};const n=t.getProgramParameter(e,t.ACTIVE_UNIFORMS);for(let i=0;i<n;++i){const r=t.getActiveUniform(e,i),s=t.getUniformLocation(e,r.name);gf(r,s,this)}}setValue(t,e,n,i){const r=this.map[e];r!==void 0&&r.setValue(t,n,i)}setOptional(t,e,n){const i=e[n];i!==void 0&&this.setValue(t,n,i)}static upload(t,e,n,i){for(let r=0,s=e.length;r!==s;++r){const o=e[r],a=n[o.id];a.needsUpdate!==!1&&o.setValue(t,a.value,i)}}static seqWithValue(t,e){const n=[];for(let i=0,r=t.length;i!==r;++i){const s=t[i];s.id in e&&n.push(s)}return n}};function oo(t,e,n){const i=t.createShader(e);return t.shaderSource(i,n),t.compileShader(i),i}var mf=37297,Ef=0;function Sf(t,e){const n=t.split(`
`),i=[],r=Math.max(e-6,0),s=Math.min(e+6,n.length);for(let o=r;o<s;o++){const a=o+1;i.push(`${a===e?">":" "} ${a}: ${n[o]}`)}return i.join(`
`)}function vf(t){const e=je.getPrimaries(je.workingColorSpace),n=je.getPrimaries(t);let i;switch(e===n?i="":e===Ti&&n===Ci?i="LinearDisplayP3ToLinearSRGB":e===Ci&&n===Ti&&(i="LinearSRGBToLinearDisplayP3"),t){case zt:case Ii:return[i,"LinearTransferOETF"];case ot:case Tr:return[i,"sRGBTransferOETF"];default:return console.warn("THREE.WebGLProgram: Unsupported color space:",t),[i,"LinearTransferOETF"]}}function Ao(t,e,n){const i=t.getShaderParameter(e,t.COMPILE_STATUS),r=t.getShaderInfoLog(e).trim();if(i&&r==="")return"";const s=/ERROR: 0:(\d+)/.exec(r);if(s){const o=parseInt(s[1]);return n.toUpperCase()+`

`+r+`

`+Sf(t.getShaderSource(e),o)}else return r}function xf(t,e){const n=vf(e);return`vec4 ${t}( vec4 value ) { return ${n[0]}( ${n[1]}( value ) ); }`}function Mf(t,e){let n;switch(e){case gA:n="Linear";break;case mA:n="Reinhard";break;case EA:n="OptimizedCineon";break;case SA:n="ACESFilmic";break;case xA:n="AgX";break;case vA:n="Custom";break;default:console.warn("THREE.WebGLProgram: Unsupported toneMapping:",e),n="Linear"}return"vec3 "+t+"( vec3 color ) { return "+n+"ToneMapping( color ); }"}function If(t){return[t.extensionDerivatives||t.envMapCubeUVHeight||t.bumpMap||t.normalMapTangentSpace||t.clearcoatNormalMap||t.flatShading||t.shaderID==="physical"?"#extension GL_OES_standard_derivatives : enable":"",(t.extensionFragDepth||t.logarithmicDepthBuffer)&&t.rendererExtensionFragDepth?"#extension GL_EXT_frag_depth : enable":"",t.extensionDrawBuffers&&t.rendererExtensionDrawBuffers?"#extension GL_EXT_draw_buffers : require":"",(t.extensionShaderTextureLOD||t.envMap||t.transmission)&&t.rendererExtensionShaderTextureLod?"#extension GL_EXT_shader_texture_lod : enable":""].filter(ti).join(`
`)}function yf(t){return[t.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":""].filter(ti).join(`
`)}function Cf(t){const e=[];for(const n in t){const i=t[n];i!==!1&&e.push("#define "+n+" "+i)}return e.join(`
`)}function Tf(t,e){const n={},i=t.getProgramParameter(e,t.ACTIVE_ATTRIBUTES);for(let r=0;r<i;r++){const s=t.getActiveAttrib(e,r),o=s.name;let a=1;s.type===t.FLOAT_MAT2&&(a=2),s.type===t.FLOAT_MAT3&&(a=3),s.type===t.FLOAT_MAT4&&(a=4),n[o]={type:s.type,location:t.getAttribLocation(e,o),locationSize:a}}return n}function ti(t){return t!==""}function lo(t,e){const n=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return t.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,n).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function co(t,e){return t.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}var Pf=/^[ \t]*#include +<([\w\d./]+)>/gm;function ts(t){return t.replace(Pf,Bf)}var bf=new Map([["encodings_fragment","colorspace_fragment"],["encodings_pars_fragment","colorspace_pars_fragment"],["output_fragment","opaque_fragment"]]);function Bf(t,e){let n=Pe[e];if(n===void 0){const i=bf.get(e);if(i!==void 0)n=Pe[i],console.warn('THREE.WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,i);else throw new Error("Can not resolve #include <"+e+">")}return ts(n)}var kf=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function uo(t){return t.replace(kf,Rf)}function Rf(t,e,n,i){let r="";for(let s=parseInt(e);s<parseInt(n);s++)r+=i.replace(/\[\s*i\s*\]/g,"[ "+s+" ]").replace(/UNROLLED_LOOP_INDEX/g,s);return r}function fo(t){let e="precision "+t.precision+` float;
precision `+t.precision+" int;";return t.precision==="highp"?e+=`
#define HIGH_PRECISION`:t.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:t.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}function _f(t){let e="SHADOWMAP_TYPE_BASIC";return t.shadowMapType===hs?e="SHADOWMAP_TYPE_PCF":t.shadowMapType===Uo?e="SHADOWMAP_TYPE_PCF_SOFT":t.shadowMapType===Nt&&(e="SHADOWMAP_TYPE_VSM"),e}function Lf(t){let e="ENVMAP_TYPE_CUBE";if(t.envMap)switch(t.envMapMode){case bn:case Bn:e="ENVMAP_TYPE_CUBE";break;case vi:e="ENVMAP_TYPE_CUBE_UV";break}return e}function wf(t){let e="ENVMAP_MODE_REFLECTION";if(t.envMap)switch(t.envMapMode){case Bn:e="ENVMAP_MODE_REFRACTION";break}return e}function Jf(t){let e="ENVMAP_BLENDING_NONE";if(t.envMap)switch(t.combine){case vs:e="ENVMAP_BLENDING_MULTIPLY";break;case hA:e="ENVMAP_BLENDING_MIX";break;case pA:e="ENVMAP_BLENDING_ADD";break}return e}function Of(t){const e=t.envMapCubeUVHeight;if(e===null)return null;const n=Math.log2(e)-2,i=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,n),112)),texelHeight:i,maxMip:n}}function Df(t,e,n,i){const r=t.getContext(),s=n.defines;let o=n.vertexShader,a=n.fragmentShader;const A=_f(n),l=Lf(n),c=wf(n),d=Jf(n),f=Of(n),p=n.isWebGL2?"":If(n),E=yf(n),g=Cf(s),h=r.createProgram();let u,v,y=n.glslVersion?"#version "+n.glslVersion+`
`:"";n.isRawShaderMaterial?(u=["#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,g].filter(ti).join(`
`),u.length>0&&(u+=`
`),v=[p,"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,g].filter(ti).join(`
`),v.length>0&&(v+=`
`)):(u=[fo(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,g,n.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",n.batching?"#define USE_BATCHING":"",n.instancing?"#define USE_INSTANCING":"",n.instancingColor?"#define USE_INSTANCING_COLOR":"",n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.map?"#define USE_MAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+c:"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.displacementMap?"#define USE_DISPLACEMENTMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.mapUv?"#define MAP_UV "+n.mapUv:"",n.alphaMapUv?"#define ALPHAMAP_UV "+n.alphaMapUv:"",n.lightMapUv?"#define LIGHTMAP_UV "+n.lightMapUv:"",n.aoMapUv?"#define AOMAP_UV "+n.aoMapUv:"",n.emissiveMapUv?"#define EMISSIVEMAP_UV "+n.emissiveMapUv:"",n.bumpMapUv?"#define BUMPMAP_UV "+n.bumpMapUv:"",n.normalMapUv?"#define NORMALMAP_UV "+n.normalMapUv:"",n.displacementMapUv?"#define DISPLACEMENTMAP_UV "+n.displacementMapUv:"",n.metalnessMapUv?"#define METALNESSMAP_UV "+n.metalnessMapUv:"",n.roughnessMapUv?"#define ROUGHNESSMAP_UV "+n.roughnessMapUv:"",n.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+n.anisotropyMapUv:"",n.clearcoatMapUv?"#define CLEARCOATMAP_UV "+n.clearcoatMapUv:"",n.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+n.clearcoatNormalMapUv:"",n.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+n.clearcoatRoughnessMapUv:"",n.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+n.iridescenceMapUv:"",n.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+n.iridescenceThicknessMapUv:"",n.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+n.sheenColorMapUv:"",n.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+n.sheenRoughnessMapUv:"",n.specularMapUv?"#define SPECULARMAP_UV "+n.specularMapUv:"",n.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+n.specularColorMapUv:"",n.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+n.specularIntensityMapUv:"",n.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+n.transmissionMapUv:"",n.thicknessMapUv?"#define THICKNESSMAP_UV "+n.thicknessMapUv:"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexColors?"#define USE_COLOR":"",n.vertexAlphas?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.flatShading?"#define FLAT_SHADED":"",n.skinning?"#define USE_SKINNING":"",n.morphTargets?"#define USE_MORPHTARGETS":"",n.morphNormals&&n.flatShading===!1?"#define USE_MORPHNORMALS":"",n.morphColors&&n.isWebGL2?"#define USE_MORPHCOLORS":"",n.morphTargetsCount>0&&n.isWebGL2?"#define MORPHTARGETS_TEXTURE":"",n.morphTargetsCount>0&&n.isWebGL2?"#define MORPHTARGETS_TEXTURE_STRIDE "+n.morphTextureStride:"",n.morphTargetsCount>0&&n.isWebGL2?"#define MORPHTARGETS_COUNT "+n.morphTargetsCount:"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+A:"",n.sizeAttenuation?"#define USE_SIZEATTENUATION":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.useLegacyLights?"#define LEGACY_LIGHTS":"",n.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",n.logarithmicDepthBuffer&&n.rendererExtensionFragDepth?"#define USE_LOGDEPTHBUF_EXT":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#if ( defined( USE_MORPHTARGETS ) && ! defined( MORPHTARGETS_TEXTURE ) )","	attribute vec3 morphTarget0;","	attribute vec3 morphTarget1;","	attribute vec3 morphTarget2;","	attribute vec3 morphTarget3;","	#ifdef USE_MORPHNORMALS","		attribute vec3 morphNormal0;","		attribute vec3 morphNormal1;","		attribute vec3 morphNormal2;","		attribute vec3 morphNormal3;","	#else","		attribute vec3 morphTarget4;","		attribute vec3 morphTarget5;","		attribute vec3 morphTarget6;","		attribute vec3 morphTarget7;","	#endif","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(ti).join(`
`),v=[p,fo(n),"#define SHADER_TYPE "+n.shaderType,"#define SHADER_NAME "+n.shaderName,g,n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.map?"#define USE_MAP":"",n.matcap?"#define USE_MATCAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+l:"",n.envMap?"#define "+c:"",n.envMap?"#define "+d:"",f?"#define CUBEUV_TEXEL_WIDTH "+f.texelWidth:"",f?"#define CUBEUV_TEXEL_HEIGHT "+f.texelHeight:"",f?"#define CUBEUV_MAX_MIP "+f.maxMip+".0":"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",n.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.anisotropy?"#define USE_ANISOTROPY":"",n.anisotropyMap?"#define USE_ANISOTROPYMAP":"",n.clearcoat?"#define USE_CLEARCOAT":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.iridescence?"#define USE_IRIDESCENCE":"",n.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",n.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",n.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaTest?"#define USE_ALPHATEST":"",n.alphaHash?"#define USE_ALPHAHASH":"",n.sheen?"#define USE_SHEEN":"",n.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.vertexTangents&&n.flatShading===!1?"#define USE_TANGENT":"",n.vertexColors||n.instancingColor?"#define USE_COLOR":"",n.vertexAlphas?"#define USE_COLOR_ALPHA":"",n.vertexUv1s?"#define USE_UV1":"",n.vertexUv2s?"#define USE_UV2":"",n.vertexUv3s?"#define USE_UV3":"",n.pointsUvs?"#define USE_POINTS_UV":"",n.gradientMap?"#define USE_GRADIENTMAP":"",n.flatShading?"#define FLAT_SHADED":"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+A:"",n.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",n.numLightProbes>0?"#define USE_LIGHT_PROBES":"",n.useLegacyLights?"#define LEGACY_LIGHTS":"",n.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",n.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",n.logarithmicDepthBuffer&&n.rendererExtensionFragDepth?"#define USE_LOGDEPTHBUF_EXT":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",n.toneMapping!==tn?"#define TONE_MAPPING":"",n.toneMapping!==tn?Pe.tonemapping_pars_fragment:"",n.toneMapping!==tn?Mf("toneMapping",n.toneMapping):"",n.dithering?"#define DITHERING":"",n.opaque?"#define OPAQUE":"",Pe.colorspace_pars_fragment,xf("linearToOutputTexel",n.outputColorSpace),n.useDepthPacking?"#define DEPTH_PACKING "+n.depthPacking:"",`
`].filter(ti).join(`
`)),o=ts(o),o=lo(o,n),o=co(o,n),a=ts(a),a=lo(a,n),a=co(a,n),o=uo(o),a=uo(a),n.isWebGL2&&n.isRawShaderMaterial!==!0&&(y=`#version 300 es
`,u=[E,"precision mediump sampler2DArray;","#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+u,v=["precision mediump sampler2DArray;","#define varying in",n.glslVersion===Aa?"":"layout(location = 0) out highp vec4 pc_fragColor;",n.glslVersion===Aa?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+v);const T=y+u+o,_=y+v+a,C=oo(r,r.VERTEX_SHADER,T),B=oo(r,r.FRAGMENT_SHADER,_);r.attachShader(h,C),r.attachShader(h,B),n.index0AttributeName!==void 0?r.bindAttribLocation(h,0,n.index0AttributeName):n.morphTargets===!0&&r.bindAttribLocation(h,0,"position"),r.linkProgram(h);function z(W){if(t.debug.checkShaderErrors){const Y=r.getProgramInfoLog(h).trim(),b=r.getShaderInfoLog(C).trim(),Q=r.getShaderInfoLog(B).trim();let G=!0,q=!0;if(r.getProgramParameter(h,r.LINK_STATUS)===!1)if(G=!1,typeof t.debug.onShaderError=="function")t.debug.onShaderError(r,h,C,B);else{const U=Ao(r,C,"vertex"),j=Ao(r,B,"fragment");console.error("THREE.WebGLProgram: Shader Error "+r.getError()+" - VALIDATE_STATUS "+r.getProgramParameter(h,r.VALIDATE_STATUS)+`

Program Info Log: `+Y+`
`+U+`
`+j)}else Y!==""?console.warn("THREE.WebGLProgram: Program Info Log:",Y):(b===""||Q==="")&&(q=!1);q&&(W.diagnostics={runnable:G,programLog:Y,vertexShader:{log:b,prefix:u},fragmentShader:{log:Q,prefix:v}})}r.deleteShader(C),r.deleteShader(B),M=new tr(r,h),I=Tf(r,h)}let M;this.getUniforms=function(){return M===void 0&&z(this),M};let I;this.getAttributes=function(){return I===void 0&&z(this),I};let H=n.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return H===!1&&(H=r.getProgramParameter(h,mf)),H},this.destroy=function(){i.releaseStatesOfProgram(this),r.deleteProgram(h),this.program=void 0},this.type=n.shaderType,this.name=n.shaderName,this.id=Ef++,this.cacheKey=e,this.usedTimes=1,this.program=h,this.vertexShader=C,this.fragmentShader=B,this}var Hf=0,Qf=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(t){const e=t.vertexShader,n=t.fragmentShader,i=this._getShaderStage(e),r=this._getShaderStage(n),s=this._getShaderCacheForMaterial(t);return s.has(i)===!1&&(s.add(i),i.usedTimes++),s.has(r)===!1&&(s.add(r),r.usedTimes++),this}remove(t){const e=this.materialCache.get(t);for(const n of e)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(t),this}getVertexShaderID(t){return this._getShaderStage(t.vertexShader).id}getFragmentShaderID(t){return this._getShaderStage(t.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(t){const e=this.materialCache;let n=e.get(t);return n===void 0&&(n=new Set,e.set(t,n)),n}_getShaderStage(t){const e=this.shaderCache;let n=e.get(t);return n===void 0&&(n=new Nf(t),e.set(t,n)),n}},Nf=class{constructor(t){this.id=Hf++,this.code=t,this.usedTimes=0}};function Gf(t,e,n,i,r,s,o){const a=new Ma,A=new Qf,l=[],c=r.isWebGL2,d=r.logarithmicDepthBuffer,f=r.vertexTextures;let p=r.precision;const E={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function g(M){return M===0?"uv":`uv${M}`}function h(M,I,H,W,Y){const b=W.fog,Q=Y.geometry,G=M.isMeshStandardMaterial?W.environment:null,q=(M.isMeshStandardMaterial?n:e).get(M.envMap||G),U=q&&q.mapping===vi?q.image.height:null,j=E[M.type];M.precision!==null&&(p=r.getMaxPrecision(M.precision),p!==M.precision&&console.warn("THREE.WebGLProgram.getParameters:",M.precision,"not supported, using",p,"instead."));const X=Q.morphAttributes.position||Q.morphAttributes.normal||Q.morphAttributes.color,ee=X!==void 0?X.length:0;let k=0;Q.morphAttributes.position!==void 0&&(k=1),Q.morphAttributes.normal!==void 0&&(k=2),Q.morphAttributes.color!==void 0&&(k=3);let Z,re,se,me;if(j){const vt=Dt[j];Z=vt.vertexShader,re=vt.fragmentShader}else Z=M.vertexShader,re=M.fragmentShader,A.update(M),se=A.getVertexShaderID(M),me=A.getFragmentShaderID(M);const xe=t.getRenderTarget(),Le=Y.isInstancedMesh===!0,Me=Y.isBatchedMesh===!0,ze=!!M.map,L=!!M.matcap,St=!!q,He=!!M.aoMap,he=!!M.lightMap,Se=!!M.bumpMap,be=!!M.normalMap,De=!!M.displacementMap,Re=!!M.emissiveMap,x=!!M.metalnessMap,m=!!M.roughnessMap,O=M.anisotropy>0,$=M.clearcoat>0,F=M.iridescence>0,K=M.sheen>0,ge=M.transmission>0,ae=O&&!!M.anisotropyMap,le=$&&!!M.clearcoatMap,fe=$&&!!M.clearcoatNormalMap,Be=$&&!!M.clearcoatRoughnessMap,V=F&&!!M.iridescenceMap,tt=F&&!!M.iridescenceThicknessMap,we=K&&!!M.sheenColorMap,ye=K&&!!M.sheenRoughnessMap,ce=!!M.specularMap,ue=!!M.specularColorMap,Je=!!M.specularIntensityMap,Ue=ge&&!!M.transmissionMap,Qe=ge&&!!M.thicknessMap,Xe=!!M.gradientMap,ne=!!M.alphaMap,P=M.alphaTest>0,ie=!!M.alphaHash,oe=!!M.extensions,ve=!!Q.attributes.uv1,pe=!!Q.attributes.uv2,Ze=!!Q.attributes.uv3;let Ye=tn;return M.toneMapped&&(xe===null||xe.isXRRenderTarget===!0)&&(Ye=t.toneMapping),{isWebGL2:c,shaderID:j,shaderType:M.type,shaderName:M.name,vertexShader:Z,fragmentShader:re,defines:M.defines,customVertexShaderID:se,customFragmentShaderID:me,isRawShaderMaterial:M.isRawShaderMaterial===!0,glslVersion:M.glslVersion,precision:p,batching:Me,instancing:Le,instancingColor:Le&&Y.instanceColor!==null,supportsVertexTextures:f,outputColorSpace:xe===null?t.outputColorSpace:xe.isXRRenderTarget===!0?xe.texture.colorSpace:zt,map:ze,matcap:L,envMap:St,envMapMode:St&&q.mapping,envMapCubeUVHeight:U,aoMap:He,lightMap:he,bumpMap:Se,normalMap:be,displacementMap:f&&De,emissiveMap:Re,normalMapObjectSpace:be&&M.normalMapType===wA,normalMapTangentSpace:be&&M.normalMapType===LA,metalnessMap:x,roughnessMap:m,anisotropy:O,anisotropyMap:ae,clearcoat:$,clearcoatMap:le,clearcoatNormalMap:fe,clearcoatRoughnessMap:Be,iridescence:F,iridescenceMap:V,iridescenceThicknessMap:tt,sheen:K,sheenColorMap:we,sheenRoughnessMap:ye,specularMap:ce,specularColorMap:ue,specularIntensityMap:Je,transmission:ge,transmissionMap:Ue,thicknessMap:Qe,gradientMap:Xe,opaque:M.transparent===!1&&M.blending===dn,alphaMap:ne,alphaTest:P,alphaHash:ie,combine:M.combine,mapUv:ze&&g(M.map.channel),aoMapUv:He&&g(M.aoMap.channel),lightMapUv:he&&g(M.lightMap.channel),bumpMapUv:Se&&g(M.bumpMap.channel),normalMapUv:be&&g(M.normalMap.channel),displacementMapUv:De&&g(M.displacementMap.channel),emissiveMapUv:Re&&g(M.emissiveMap.channel),metalnessMapUv:x&&g(M.metalnessMap.channel),roughnessMapUv:m&&g(M.roughnessMap.channel),anisotropyMapUv:ae&&g(M.anisotropyMap.channel),clearcoatMapUv:le&&g(M.clearcoatMap.channel),clearcoatNormalMapUv:fe&&g(M.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Be&&g(M.clearcoatRoughnessMap.channel),iridescenceMapUv:V&&g(M.iridescenceMap.channel),iridescenceThicknessMapUv:tt&&g(M.iridescenceThicknessMap.channel),sheenColorMapUv:we&&g(M.sheenColorMap.channel),sheenRoughnessMapUv:ye&&g(M.sheenRoughnessMap.channel),specularMapUv:ce&&g(M.specularMap.channel),specularColorMapUv:ue&&g(M.specularColorMap.channel),specularIntensityMapUv:Je&&g(M.specularIntensityMap.channel),transmissionMapUv:Ue&&g(M.transmissionMap.channel),thicknessMapUv:Qe&&g(M.thicknessMap.channel),alphaMapUv:ne&&g(M.alphaMap.channel),vertexTangents:!!Q.attributes.tangent&&(be||O),vertexColors:M.vertexColors,vertexAlphas:M.vertexColors===!0&&!!Q.attributes.color&&Q.attributes.color.itemSize===4,vertexUv1s:ve,vertexUv2s:pe,vertexUv3s:Ze,pointsUvs:Y.isPoints===!0&&!!Q.attributes.uv&&(ze||ne),fog:!!b,useFog:M.fog===!0,fogExp2:b&&b.isFogExp2,flatShading:M.flatShading===!0,sizeAttenuation:M.sizeAttenuation===!0,logarithmicDepthBuffer:d,skinning:Y.isSkinnedMesh===!0,morphTargets:Q.morphAttributes.position!==void 0,morphNormals:Q.morphAttributes.normal!==void 0,morphColors:Q.morphAttributes.color!==void 0,morphTargetsCount:ee,morphTextureStride:k,numDirLights:I.directional.length,numPointLights:I.point.length,numSpotLights:I.spot.length,numSpotLightMaps:I.spotLightMap.length,numRectAreaLights:I.rectArea.length,numHemiLights:I.hemi.length,numDirLightShadows:I.directionalShadowMap.length,numPointLightShadows:I.pointShadowMap.length,numSpotLightShadows:I.spotShadowMap.length,numSpotLightShadowsWithMaps:I.numSpotLightShadowsWithMaps,numLightProbes:I.numLightProbes,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:M.dithering,shadowMapEnabled:t.shadowMap.enabled&&H.length>0,shadowMapType:t.shadowMap.type,toneMapping:Ye,useLegacyLights:t._useLegacyLights,decodeVideoTexture:ze&&M.map.isVideoTexture===!0&&je.getTransfer(M.map.colorSpace)===We,premultipliedAlpha:M.premultipliedAlpha,doubleSided:M.side===Gt,flipSided:M.side===ut,useDepthPacking:M.depthPacking>=0,depthPacking:M.depthPacking||0,index0AttributeName:M.index0AttributeName,extensionDerivatives:oe&&M.extensions.derivatives===!0,extensionFragDepth:oe&&M.extensions.fragDepth===!0,extensionDrawBuffers:oe&&M.extensions.drawBuffers===!0,extensionShaderTextureLOD:oe&&M.extensions.shaderTextureLOD===!0,extensionClipCullDistance:oe&&M.extensions.clipCullDistance&&i.has("WEBGL_clip_cull_distance"),rendererExtensionFragDepth:c||i.has("EXT_frag_depth"),rendererExtensionDrawBuffers:c||i.has("WEBGL_draw_buffers"),rendererExtensionShaderTextureLod:c||i.has("EXT_shader_texture_lod"),rendererExtensionParallelShaderCompile:i.has("KHR_parallel_shader_compile"),customProgramCacheKey:M.customProgramCacheKey()}}function u(M){const I=[];if(M.shaderID?I.push(M.shaderID):(I.push(M.customVertexShaderID),I.push(M.customFragmentShaderID)),M.defines!==void 0)for(const H in M.defines)I.push(H),I.push(M.defines[H]);return M.isRawShaderMaterial===!1&&(v(I,M),y(I,M),I.push(t.outputColorSpace)),I.push(M.customProgramCacheKey),I.join()}function v(M,I){M.push(I.precision),M.push(I.outputColorSpace),M.push(I.envMapMode),M.push(I.envMapCubeUVHeight),M.push(I.mapUv),M.push(I.alphaMapUv),M.push(I.lightMapUv),M.push(I.aoMapUv),M.push(I.bumpMapUv),M.push(I.normalMapUv),M.push(I.displacementMapUv),M.push(I.emissiveMapUv),M.push(I.metalnessMapUv),M.push(I.roughnessMapUv),M.push(I.anisotropyMapUv),M.push(I.clearcoatMapUv),M.push(I.clearcoatNormalMapUv),M.push(I.clearcoatRoughnessMapUv),M.push(I.iridescenceMapUv),M.push(I.iridescenceThicknessMapUv),M.push(I.sheenColorMapUv),M.push(I.sheenRoughnessMapUv),M.push(I.specularMapUv),M.push(I.specularColorMapUv),M.push(I.specularIntensityMapUv),M.push(I.transmissionMapUv),M.push(I.thicknessMapUv),M.push(I.combine),M.push(I.fogExp2),M.push(I.sizeAttenuation),M.push(I.morphTargetsCount),M.push(I.morphAttributeCount),M.push(I.numDirLights),M.push(I.numPointLights),M.push(I.numSpotLights),M.push(I.numSpotLightMaps),M.push(I.numHemiLights),M.push(I.numRectAreaLights),M.push(I.numDirLightShadows),M.push(I.numPointLightShadows),M.push(I.numSpotLightShadows),M.push(I.numSpotLightShadowsWithMaps),M.push(I.numLightProbes),M.push(I.shadowMapType),M.push(I.toneMapping),M.push(I.numClippingPlanes),M.push(I.numClipIntersection),M.push(I.depthPacking)}function y(M,I){a.disableAll(),I.isWebGL2&&a.enable(0),I.supportsVertexTextures&&a.enable(1),I.instancing&&a.enable(2),I.instancingColor&&a.enable(3),I.matcap&&a.enable(4),I.envMap&&a.enable(5),I.normalMapObjectSpace&&a.enable(6),I.normalMapTangentSpace&&a.enable(7),I.clearcoat&&a.enable(8),I.iridescence&&a.enable(9),I.alphaTest&&a.enable(10),I.vertexColors&&a.enable(11),I.vertexAlphas&&a.enable(12),I.vertexUv1s&&a.enable(13),I.vertexUv2s&&a.enable(14),I.vertexUv3s&&a.enable(15),I.vertexTangents&&a.enable(16),I.anisotropy&&a.enable(17),I.alphaHash&&a.enable(18),I.batching&&a.enable(19),M.push(a.mask),a.disableAll(),I.fog&&a.enable(0),I.useFog&&a.enable(1),I.flatShading&&a.enable(2),I.logarithmicDepthBuffer&&a.enable(3),I.skinning&&a.enable(4),I.morphTargets&&a.enable(5),I.morphNormals&&a.enable(6),I.morphColors&&a.enable(7),I.premultipliedAlpha&&a.enable(8),I.shadowMapEnabled&&a.enable(9),I.useLegacyLights&&a.enable(10),I.doubleSided&&a.enable(11),I.flipSided&&a.enable(12),I.useDepthPacking&&a.enable(13),I.dithering&&a.enable(14),I.transmission&&a.enable(15),I.sheen&&a.enable(16),I.opaque&&a.enable(17),I.pointsUvs&&a.enable(18),I.decodeVideoTexture&&a.enable(19),M.push(a.mask)}function T(M){const I=E[M.type];let H;if(I){const W=Dt[I];H=vl.clone(W.uniforms)}else H=M.uniforms;return H}function _(M,I){let H;for(let W=0,Y=l.length;W<Y;W++){const b=l[W];if(b.cacheKey===I){H=b,++H.usedTimes;break}}return H===void 0&&(H=new Df(t,I,M,s),l.push(H)),H}function C(M){if(--M.usedTimes===0){const I=l.indexOf(M);l[I]=l[l.length-1],l.pop(),M.destroy()}}function B(M){A.remove(M)}function z(){A.dispose()}return{getParameters:h,getProgramCacheKey:u,getUniforms:T,acquireProgram:_,releaseProgram:C,releaseShaderCache:B,programs:l,dispose:z}}function zf(){let t=new WeakMap;function e(s){let o=t.get(s);return o===void 0&&(o={},t.set(s,o)),o}function n(s){t.delete(s)}function i(s,o,a){t.get(s)[o]=a}function r(){t=new WeakMap}return{get:e,remove:n,update:i,dispose:r}}function jf(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.material.id!==e.material.id?t.material.id-e.material.id:t.z!==e.z?t.z-e.z:t.id-e.id}function ho(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.z!==e.z?e.z-t.z:t.id-e.id}function po(){const t=[];let e=0;const n=[],i=[],r=[];function s(){e=0,n.length=0,i.length=0,r.length=0}function o(d,f,p,E,g,h){let u=t[e];return u===void 0?(u={id:d.id,object:d,geometry:f,material:p,groupOrder:E,renderOrder:d.renderOrder,z:g,group:h},t[e]=u):(u.id=d.id,u.object=d,u.geometry=f,u.material=p,u.groupOrder=E,u.renderOrder=d.renderOrder,u.z=g,u.group=h),e++,u}function a(d,f,p,E,g,h){const u=o(d,f,p,E,g,h);p.transmission>0?i.push(u):p.transparent===!0?r.push(u):n.push(u)}function A(d,f,p,E,g,h){const u=o(d,f,p,E,g,h);p.transmission>0?i.unshift(u):p.transparent===!0?r.unshift(u):n.unshift(u)}function l(d,f){n.length>1&&n.sort(d||jf),i.length>1&&i.sort(f||ho),r.length>1&&r.sort(f||ho)}function c(){for(let d=e,f=t.length;d<f;d++){const p=t[d];if(p.id===null)break;p.id=null,p.object=null,p.geometry=null,p.material=null,p.group=null}}return{opaque:n,transmissive:i,transparent:r,init:s,push:a,unshift:A,finish:c,sort:l}}function Uf(){let t=new WeakMap;function e(i,r){const s=t.get(i);let o;return s===void 0?(o=new po,t.set(i,[o])):r>=s.length?(o=new po,s.push(o)):o=s[r],o}function n(){t=new WeakMap}return{get:e,dispose:n}}function Xf(){const t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"DirectionalLight":n={direction:new J,color:new Ne};break;case"SpotLight":n={position:new J,direction:new J,color:new Ne,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":n={position:new J,color:new Ne,distance:0,decay:0};break;case"HemisphereLight":n={direction:new J,skyColor:new Ne,groundColor:new Ne};break;case"RectAreaLight":n={color:new Ne,position:new J,halfWidth:new J,halfHeight:new J};break}return t[e.id]=n,n}}}function Zf(){const t={};return{get:function(e){if(t[e.id]!==void 0)return t[e.id];let n;switch(e.type){case"DirectionalLight":n={shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Fe};break;case"SpotLight":n={shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Fe};break;case"PointLight":n={shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Fe,shadowCameraNear:1,shadowCameraFar:1e3};break}return t[e.id]=n,n}}}var Ff=0;function Wf(t,e){return(e.castShadow?2:0)-(t.castShadow?2:0)+(e.map?1:0)-(t.map?1:0)}function Yf(t,e){const n=new Xf,i=Zf(),r={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)r.probe.push(new J);const s=new J,o=new gt,a=new gt;function A(c,d){let f=0,p=0,E=0;for(let W=0;W<9;W++)r.probe[W].set(0,0,0);let g=0,h=0,u=0,v=0,y=0,T=0,_=0,C=0,B=0,z=0,M=0;c.sort(Wf);const I=d===!0?Math.PI:1;for(let W=0,Y=c.length;W<Y;W++){const b=c[W],Q=b.color,G=b.intensity,q=b.distance,U=b.shadow&&b.shadow.map?b.shadow.map.texture:null;if(b.isAmbientLight)f+=Q.r*G*I,p+=Q.g*G*I,E+=Q.b*G*I;else if(b.isLightProbe){for(let j=0;j<9;j++)r.probe[j].addScaledVector(b.sh.coefficients[j],G);M++}else if(b.isDirectionalLight){const j=n.get(b);if(j.color.copy(b.color).multiplyScalar(b.intensity*I),b.castShadow){const X=b.shadow,ee=i.get(b);ee.shadowBias=X.bias,ee.shadowNormalBias=X.normalBias,ee.shadowRadius=X.radius,ee.shadowMapSize=X.mapSize,r.directionalShadow[g]=ee,r.directionalShadowMap[g]=U,r.directionalShadowMatrix[g]=b.shadow.matrix,T++}r.directional[g]=j,g++}else if(b.isSpotLight){const j=n.get(b);j.position.setFromMatrixPosition(b.matrixWorld),j.color.copy(Q).multiplyScalar(G*I),j.distance=q,j.coneCos=Math.cos(b.angle),j.penumbraCos=Math.cos(b.angle*(1-b.penumbra)),j.decay=b.decay,r.spot[u]=j;const X=b.shadow;if(b.map&&(r.spotLightMap[B]=b.map,B++,X.updateMatrices(b),b.castShadow&&z++),r.spotLightMatrix[u]=X.matrix,b.castShadow){const ee=i.get(b);ee.shadowBias=X.bias,ee.shadowNormalBias=X.normalBias,ee.shadowRadius=X.radius,ee.shadowMapSize=X.mapSize,r.spotShadow[u]=ee,r.spotShadowMap[u]=U,C++}u++}else if(b.isRectAreaLight){const j=n.get(b);j.color.copy(Q).multiplyScalar(G),j.halfWidth.set(b.width*.5,0,0),j.halfHeight.set(0,b.height*.5,0),r.rectArea[v]=j,v++}else if(b.isPointLight){const j=n.get(b);if(j.color.copy(b.color).multiplyScalar(b.intensity*I),j.distance=b.distance,j.decay=b.decay,b.castShadow){const X=b.shadow,ee=i.get(b);ee.shadowBias=X.bias,ee.shadowNormalBias=X.normalBias,ee.shadowRadius=X.radius,ee.shadowMapSize=X.mapSize,ee.shadowCameraNear=X.camera.near,ee.shadowCameraFar=X.camera.far,r.pointShadow[h]=ee,r.pointShadowMap[h]=U,r.pointShadowMatrix[h]=b.shadow.matrix,_++}r.point[h]=j,h++}else if(b.isHemisphereLight){const j=n.get(b);j.skyColor.copy(b.color).multiplyScalar(G*I),j.groundColor.copy(b.groundColor).multiplyScalar(G*I),r.hemi[y]=j,y++}}v>0&&(e.isWebGL2?t.has("OES_texture_float_linear")===!0?(r.rectAreaLTC1=te.LTC_FLOAT_1,r.rectAreaLTC2=te.LTC_FLOAT_2):(r.rectAreaLTC1=te.LTC_HALF_1,r.rectAreaLTC2=te.LTC_HALF_2):t.has("OES_texture_float_linear")===!0?(r.rectAreaLTC1=te.LTC_FLOAT_1,r.rectAreaLTC2=te.LTC_FLOAT_2):t.has("OES_texture_half_float_linear")===!0?(r.rectAreaLTC1=te.LTC_HALF_1,r.rectAreaLTC2=te.LTC_HALF_2):console.error("THREE.WebGLRenderer: Unable to use RectAreaLight. Missing WebGL extensions.")),r.ambient[0]=f,r.ambient[1]=p,r.ambient[2]=E;const H=r.hash;(H.directionalLength!==g||H.pointLength!==h||H.spotLength!==u||H.rectAreaLength!==v||H.hemiLength!==y||H.numDirectionalShadows!==T||H.numPointShadows!==_||H.numSpotShadows!==C||H.numSpotMaps!==B||H.numLightProbes!==M)&&(r.directional.length=g,r.spot.length=u,r.rectArea.length=v,r.point.length=h,r.hemi.length=y,r.directionalShadow.length=T,r.directionalShadowMap.length=T,r.pointShadow.length=_,r.pointShadowMap.length=_,r.spotShadow.length=C,r.spotShadowMap.length=C,r.directionalShadowMatrix.length=T,r.pointShadowMatrix.length=_,r.spotLightMatrix.length=C+B-z,r.spotLightMap.length=B,r.numSpotLightShadowsWithMaps=z,r.numLightProbes=M,H.directionalLength=g,H.pointLength=h,H.spotLength=u,H.rectAreaLength=v,H.hemiLength=y,H.numDirectionalShadows=T,H.numPointShadows=_,H.numSpotShadows=C,H.numSpotMaps=B,H.numLightProbes=M,r.version=Ff++)}function l(c,d){let f=0,p=0,E=0,g=0,h=0;const u=d.matrixWorldInverse;for(let v=0,y=c.length;v<y;v++){const T=c[v];if(T.isDirectionalLight){const _=r.directional[f];_.direction.setFromMatrixPosition(T.matrixWorld),s.setFromMatrixPosition(T.target.matrixWorld),_.direction.sub(s),_.direction.transformDirection(u),f++}else if(T.isSpotLight){const _=r.spot[E];_.position.setFromMatrixPosition(T.matrixWorld),_.position.applyMatrix4(u),_.direction.setFromMatrixPosition(T.matrixWorld),s.setFromMatrixPosition(T.target.matrixWorld),_.direction.sub(s),_.direction.transformDirection(u),E++}else if(T.isRectAreaLight){const _=r.rectArea[g];_.position.setFromMatrixPosition(T.matrixWorld),_.position.applyMatrix4(u),a.identity(),o.copy(T.matrixWorld),o.premultiply(u),a.extractRotation(o),_.halfWidth.set(T.width*.5,0,0),_.halfHeight.set(0,T.height*.5,0),_.halfWidth.applyMatrix4(a),_.halfHeight.applyMatrix4(a),g++}else if(T.isPointLight){const _=r.point[p];_.position.setFromMatrixPosition(T.matrixWorld),_.position.applyMatrix4(u),p++}else if(T.isHemisphereLight){const _=r.hemi[h];_.direction.setFromMatrixPosition(T.matrixWorld),_.direction.transformDirection(u),h++}}}return{setup:A,setupView:l,state:r}}function go(t,e){const n=new Yf(t,e),i=[],r=[];function s(){i.length=0,r.length=0}function o(d){i.push(d)}function a(d){r.push(d)}function A(d){n.setup(i,d)}function l(d){n.setupView(i,d)}return{init:s,state:{lightsArray:i,shadowsArray:r,lights:n},setupLights:A,setupLightsView:l,pushLight:o,pushShadow:a}}function Vf(t,e){let n=new WeakMap;function i(s,o=0){const a=n.get(s);let A;return a===void 0?(A=new go(t,e),n.set(s,[A])):o>=a.length?(A=new go(t,e),a.push(A)):A=a[o],A}function r(){n=new WeakMap}return{get:i,dispose:r}}var qf=class extends fi{constructor(t){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=RA,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(t)}copy(t){return super.copy(t),this.depthPacking=t.depthPacking,this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this}},Kf=class extends fi{constructor(t){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(t)}copy(t){return super.copy(t),this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this}},$f=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,eh=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function th(t,e,n){let i=new Na;const r=new Fe,s=new Fe,o=new pt,a=new qf({depthPacking:_A}),A=new Kf,l={},c=n.maxTextureSize,d={[$t]:ut,[ut]:$t,[Gt]:Gt},f=new Vt({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Fe},radius:{value:4}},vertexShader:$f,fragmentShader:eh}),p=f.clone();p.defines.HORIZONTAL_PASS=1;const E=new un;E.setAttribute("position",new Ot(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const g=new Yt(E,f),h=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=hs;let u=this.type;this.render=function(C,B,z){if(h.enabled===!1||h.autoUpdate===!1&&h.needsUpdate===!1||C.length===0)return;const M=t.getRenderTarget(),I=t.getActiveCubeFace(),H=t.getActiveMipmapLevel(),W=t.state;W.setBlending(en),W.buffers.color.setClear(1,1,1,1),W.buffers.depth.setTest(!0),W.setScissorTest(!1);const Y=u!==Nt&&this.type===Nt,b=u===Nt&&this.type!==Nt;for(let Q=0,G=C.length;Q<G;Q++){const q=C[Q],U=q.shadow;if(U===void 0){console.warn("THREE.WebGLShadowMap:",q,"has no shadow.");continue}if(U.autoUpdate===!1&&U.needsUpdate===!1)continue;r.copy(U.mapSize);const j=U.getFrameExtents();if(r.multiply(j),s.copy(U.mapSize),(r.x>c||r.y>c)&&(r.x>c&&(s.x=Math.floor(c/j.x),r.x=s.x*j.x,U.mapSize.x=s.x),r.y>c&&(s.y=Math.floor(c/j.y),r.y=s.y*j.y,U.mapSize.y=s.y)),U.map===null||Y===!0||b===!0){const ee=this.type!==Nt?{minFilter:dt,magFilter:dt}:{};U.map!==null&&U.map.dispose(),U.map=new mn(r.x,r.y,ee),U.map.texture.name=q.name+".shadowMap",U.camera.updateProjectionMatrix()}t.setRenderTarget(U.map),t.clear();const X=U.getViewportCount();for(let ee=0;ee<X;ee++){const k=U.getViewport(ee);o.set(s.x*k.x,s.y*k.y,s.x*k.z,s.y*k.w),W.viewport(o),U.updateMatrices(q,ee),i=U.getFrustum(),T(B,z,U.camera,q,this.type)}U.isPointLightShadow!==!0&&this.type===Nt&&v(U,z),U.needsUpdate=!1}u=this.type,h.needsUpdate=!1,t.setRenderTarget(M,I,H)};function v(C,B){const z=e.update(g);f.defines.VSM_SAMPLES!==C.blurSamples&&(f.defines.VSM_SAMPLES=C.blurSamples,p.defines.VSM_SAMPLES=C.blurSamples,f.needsUpdate=!0,p.needsUpdate=!0),C.mapPass===null&&(C.mapPass=new mn(r.x,r.y)),f.uniforms.shadow_pass.value=C.map.texture,f.uniforms.resolution.value=C.mapSize,f.uniforms.radius.value=C.radius,t.setRenderTarget(C.mapPass),t.clear(),t.renderBufferDirect(B,null,z,f,g,null),p.uniforms.shadow_pass.value=C.mapPass.texture,p.uniforms.resolution.value=C.mapSize,p.uniforms.radius.value=C.radius,t.setRenderTarget(C.map),t.clear(),t.renderBufferDirect(B,null,z,p,g,null)}function y(C,B,z,M){let I=null;const H=z.isPointLight===!0?C.customDistanceMaterial:C.customDepthMaterial;if(H!==void 0)I=H;else if(I=z.isPointLight===!0?A:a,t.localClippingEnabled&&B.clipShadows===!0&&Array.isArray(B.clippingPlanes)&&B.clippingPlanes.length!==0||B.displacementMap&&B.displacementScale!==0||B.alphaMap&&B.alphaTest>0||B.map&&B.alphaTest>0){const W=I.uuid,Y=B.uuid;let b=l[W];b===void 0&&(b={},l[W]=b);let Q=b[Y];Q===void 0&&(Q=I.clone(),b[Y]=Q,B.addEventListener("dispose",_)),I=Q}if(I.visible=B.visible,I.wireframe=B.wireframe,M===Nt?I.side=B.shadowSide!==null?B.shadowSide:B.side:I.side=B.shadowSide!==null?B.shadowSide:d[B.side],I.alphaMap=B.alphaMap,I.alphaTest=B.alphaTest,I.map=B.map,I.clipShadows=B.clipShadows,I.clippingPlanes=B.clippingPlanes,I.clipIntersection=B.clipIntersection,I.displacementMap=B.displacementMap,I.displacementScale=B.displacementScale,I.displacementBias=B.displacementBias,I.wireframeLinewidth=B.wireframeLinewidth,I.linewidth=B.linewidth,z.isPointLight===!0&&I.isMeshDistanceMaterial===!0){const W=t.properties.get(I);W.light=z}return I}function T(C,B,z,M,I){if(C.visible===!1)return;if(C.layers.test(B.layers)&&(C.isMesh||C.isLine||C.isPoints)&&(C.castShadow||C.receiveShadow&&I===Nt)&&(!C.frustumCulled||i.intersectsObject(C))){C.modelViewMatrix.multiplyMatrices(z.matrixWorldInverse,C.matrixWorld);const Y=e.update(C),b=C.material;if(Array.isArray(b)){const Q=Y.groups;for(let G=0,q=Q.length;G<q;G++){const U=Q[G],j=b[U.materialIndex];if(j&&j.visible){const X=y(C,j,M,I);C.onBeforeShadow(t,C,B,z,Y,X,U),t.renderBufferDirect(z,null,Y,X,C,U),C.onAfterShadow(t,C,B,z,Y,X,U)}}}else if(b.visible){const Q=y(C,b,M,I);C.onBeforeShadow(t,C,B,z,Y,Q,null),t.renderBufferDirect(z,null,Y,Q,C,null),C.onAfterShadow(t,C,B,z,Y,Q,null)}}const W=C.children;for(let Y=0,b=W.length;Y<b;Y++)T(W[Y],B,z,M,I)}function _(C){C.target.removeEventListener("dispose",_);for(const z in l){const M=l[z],I=C.target.uuid;I in M&&(M[I].dispose(),delete M[I])}}}function nh(t,e,n){const i=n.isWebGL2;function r(){let P=!1;const ie=new pt;let oe=null;const ve=new pt(0,0,0,0);return{setMask:function(pe){oe!==pe&&!P&&(t.colorMask(pe,pe,pe,pe),oe=pe)},setLocked:function(pe){P=pe},setClear:function(pe,Ze,Ye,rt,vt){vt===!0&&(pe*=rt,Ze*=rt,Ye*=rt),ie.set(pe,Ze,Ye,rt),ve.equals(ie)===!1&&(t.clearColor(pe,Ze,Ye,rt),ve.copy(ie))},reset:function(){P=!1,oe=null,ve.set(-1,0,0,0)}}}function s(){let P=!1,ie=null,oe=null,ve=null;return{setTest:function(pe){pe?Me(t.DEPTH_TEST):ze(t.DEPTH_TEST)},setMask:function(pe){ie!==pe&&!P&&(t.depthMask(pe),ie=pe)},setFunc:function(pe){if(oe!==pe){switch(pe){case oA:t.depthFunc(t.NEVER);break;case AA:t.depthFunc(t.ALWAYS);break;case lA:t.depthFunc(t.LESS);break;case Si:t.depthFunc(t.LEQUAL);break;case cA:t.depthFunc(t.EQUAL);break;case uA:t.depthFunc(t.GEQUAL);break;case dA:t.depthFunc(t.GREATER);break;case fA:t.depthFunc(t.NOTEQUAL);break;default:t.depthFunc(t.LEQUAL)}oe=pe}},setLocked:function(pe){P=pe},setClear:function(pe){ve!==pe&&(t.clearDepth(pe),ve=pe)},reset:function(){P=!1,ie=null,oe=null,ve=null}}}function o(){let P=!1,ie=null,oe=null,ve=null,pe=null,Ze=null,Ye=null,rt=null,vt=null;return{setTest:function(Ge){P||(Ge?Me(t.STENCIL_TEST):ze(t.STENCIL_TEST))},setMask:function(Ge){ie!==Ge&&!P&&(t.stencilMask(Ge),ie=Ge)},setFunc:function(Ge,Ht,Qt){(oe!==Ge||ve!==Ht||pe!==Qt)&&(t.stencilFunc(Ge,Ht,Qt),oe=Ge,ve=Ht,pe=Qt)},setOp:function(Ge,Ht,Qt){(Ze!==Ge||Ye!==Ht||rt!==Qt)&&(t.stencilOp(Ge,Ht,Qt),Ze=Ge,Ye=Ht,rt=Qt)},setLocked:function(Ge){P=Ge},setClear:function(Ge){vt!==Ge&&(t.clearStencil(Ge),vt=Ge)},reset:function(){P=!1,ie=null,oe=null,ve=null,pe=null,Ze=null,Ye=null,rt=null,vt=null}}}const a=new r,A=new s,l=new o,c=new WeakMap,d=new WeakMap;let f={},p={},E=new WeakMap,g=[],h=null,u=!1,v=null,y=null,T=null,_=null,C=null,B=null,z=null,M=new Ne(0,0,0),I=0,H=!1,W=null,Y=null,b=null,Q=null,G=null;const q=t.getParameter(t.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let U=!1,j=0;const X=t.getParameter(t.VERSION);X.indexOf("WebGL")!==-1?(j=parseFloat(/^WebGL (\d)/.exec(X)[1]),U=j>=1):X.indexOf("OpenGL ES")!==-1&&(j=parseFloat(/^OpenGL ES (\d)/.exec(X)[1]),U=j>=2);let ee=null,k={};const Z=t.getParameter(t.SCISSOR_BOX),re=t.getParameter(t.VIEWPORT),se=new pt().fromArray(Z),me=new pt().fromArray(re);function xe(P,ie,oe,ve){const pe=new Uint8Array(4),Ze=t.createTexture();t.bindTexture(P,Ze),t.texParameteri(P,t.TEXTURE_MIN_FILTER,t.NEAREST),t.texParameteri(P,t.TEXTURE_MAG_FILTER,t.NEAREST);for(let Ye=0;Ye<oe;Ye++)i&&(P===t.TEXTURE_3D||P===t.TEXTURE_2D_ARRAY)?t.texImage3D(ie,0,t.RGBA,1,1,ve,0,t.RGBA,t.UNSIGNED_BYTE,pe):t.texImage2D(ie+Ye,0,t.RGBA,1,1,0,t.RGBA,t.UNSIGNED_BYTE,pe);return Ze}const Le={};Le[t.TEXTURE_2D]=xe(t.TEXTURE_2D,t.TEXTURE_2D,1),Le[t.TEXTURE_CUBE_MAP]=xe(t.TEXTURE_CUBE_MAP,t.TEXTURE_CUBE_MAP_POSITIVE_X,6),i&&(Le[t.TEXTURE_2D_ARRAY]=xe(t.TEXTURE_2D_ARRAY,t.TEXTURE_2D_ARRAY,1,1),Le[t.TEXTURE_3D]=xe(t.TEXTURE_3D,t.TEXTURE_3D,1,1)),a.setClear(0,0,0,1),A.setClear(1),l.setClear(0),Me(t.DEPTH_TEST),A.setFunc(Si),Re(!1),x(fs),Me(t.CULL_FACE),be(en);function Me(P){f[P]!==!0&&(t.enable(P),f[P]=!0)}function ze(P){f[P]!==!1&&(t.disable(P),f[P]=!1)}function L(P,ie){return p[P]!==ie?(t.bindFramebuffer(P,ie),p[P]=ie,i&&(P===t.DRAW_FRAMEBUFFER&&(p[t.FRAMEBUFFER]=ie),P===t.FRAMEBUFFER&&(p[t.DRAW_FRAMEBUFFER]=ie)),!0):!1}function St(P,ie){let oe=g,ve=!1;if(P)if(oe=E.get(ie),oe===void 0&&(oe=[],E.set(ie,oe)),P.isWebGLMultipleRenderTargets){const pe=P.texture;if(oe.length!==pe.length||oe[0]!==t.COLOR_ATTACHMENT0){for(let Ze=0,Ye=pe.length;Ze<Ye;Ze++)oe[Ze]=t.COLOR_ATTACHMENT0+Ze;oe.length=pe.length,ve=!0}}else oe[0]!==t.COLOR_ATTACHMENT0&&(oe[0]=t.COLOR_ATTACHMENT0,ve=!0);else oe[0]!==t.BACK&&(oe[0]=t.BACK,ve=!0);ve&&(n.isWebGL2?t.drawBuffers(oe):e.get("WEBGL_draw_buffers").drawBuffersWEBGL(oe))}function He(P){return h!==P?(t.useProgram(P),h=P,!0):!1}const he={[fn]:t.FUNC_ADD,[Zo]:t.FUNC_SUBTRACT,[Fo]:t.FUNC_REVERSE_SUBTRACT};if(i)he[Es]=t.MIN,he[Ss]=t.MAX;else{const P=e.get("EXT_blend_minmax");P!==null&&(he[Es]=P.MIN_EXT,he[Ss]=P.MAX_EXT)}const Se={[Wo]:t.ZERO,[Yo]:t.ONE,[Vo]:t.SRC_COLOR,[dr]:t.SRC_ALPHA,[nA]:t.SRC_ALPHA_SATURATE,[eA]:t.DST_COLOR,[Ko]:t.DST_ALPHA,[qo]:t.ONE_MINUS_SRC_COLOR,[fr]:t.ONE_MINUS_SRC_ALPHA,[tA]:t.ONE_MINUS_DST_COLOR,[$o]:t.ONE_MINUS_DST_ALPHA,[iA]:t.CONSTANT_COLOR,[rA]:t.ONE_MINUS_CONSTANT_COLOR,[sA]:t.CONSTANT_ALPHA,[aA]:t.ONE_MINUS_CONSTANT_ALPHA};function be(P,ie,oe,ve,pe,Ze,Ye,rt,vt,Ge){if(P===en){u===!0&&(ze(t.BLEND),u=!1);return}if(u===!1&&(Me(t.BLEND),u=!0),P!==Xo){if(P!==v||Ge!==H){if((y!==fn||C!==fn)&&(t.blendEquation(t.FUNC_ADD),y=fn,C=fn),Ge)switch(P){case dn:t.blendFuncSeparate(t.ONE,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case ps:t.blendFunc(t.ONE,t.ONE);break;case gs:t.blendFuncSeparate(t.ZERO,t.ONE_MINUS_SRC_COLOR,t.ZERO,t.ONE);break;case ms:t.blendFuncSeparate(t.ZERO,t.SRC_COLOR,t.ZERO,t.SRC_ALPHA);break;default:console.error("THREE.WebGLState: Invalid blending: ",P);break}else switch(P){case dn:t.blendFuncSeparate(t.SRC_ALPHA,t.ONE_MINUS_SRC_ALPHA,t.ONE,t.ONE_MINUS_SRC_ALPHA);break;case ps:t.blendFunc(t.SRC_ALPHA,t.ONE);break;case gs:t.blendFuncSeparate(t.ZERO,t.ONE_MINUS_SRC_COLOR,t.ZERO,t.ONE);break;case ms:t.blendFunc(t.ZERO,t.SRC_COLOR);break;default:console.error("THREE.WebGLState: Invalid blending: ",P);break}T=null,_=null,B=null,z=null,M.set(0,0,0),I=0,v=P,H=Ge}return}pe=pe||ie,Ze=Ze||oe,Ye=Ye||ve,(ie!==y||pe!==C)&&(t.blendEquationSeparate(he[ie],he[pe]),y=ie,C=pe),(oe!==T||ve!==_||Ze!==B||Ye!==z)&&(t.blendFuncSeparate(Se[oe],Se[ve],Se[Ze],Se[Ye]),T=oe,_=ve,B=Ze,z=Ye),(rt.equals(M)===!1||vt!==I)&&(t.blendColor(rt.r,rt.g,rt.b,vt),M.copy(rt),I=vt),v=P,H=!1}function De(P,ie){P.side===Gt?ze(t.CULL_FACE):Me(t.CULL_FACE);let oe=P.side===ut;ie&&(oe=!oe),Re(oe),P.blending===dn&&P.transparent===!1?be(en):be(P.blending,P.blendEquation,P.blendSrc,P.blendDst,P.blendEquationAlpha,P.blendSrcAlpha,P.blendDstAlpha,P.blendColor,P.blendAlpha,P.premultipliedAlpha),A.setFunc(P.depthFunc),A.setTest(P.depthTest),A.setMask(P.depthWrite),a.setMask(P.colorWrite);const ve=P.stencilWrite;l.setTest(ve),ve&&(l.setMask(P.stencilWriteMask),l.setFunc(P.stencilFunc,P.stencilRef,P.stencilFuncMask),l.setOp(P.stencilFail,P.stencilZFail,P.stencilZPass)),O(P.polygonOffset,P.polygonOffsetFactor,P.polygonOffsetUnits),P.alphaToCoverage===!0?Me(t.SAMPLE_ALPHA_TO_COVERAGE):ze(t.SAMPLE_ALPHA_TO_COVERAGE)}function Re(P){W!==P&&(P?t.frontFace(t.CW):t.frontFace(t.CCW),W=P)}function x(P){P!==zo?(Me(t.CULL_FACE),P!==Y&&(P===fs?t.cullFace(t.BACK):P===jo?t.cullFace(t.FRONT):t.cullFace(t.FRONT_AND_BACK))):ze(t.CULL_FACE),Y=P}function m(P){P!==b&&(U&&t.lineWidth(P),b=P)}function O(P,ie,oe){P?(Me(t.POLYGON_OFFSET_FILL),(Q!==ie||G!==oe)&&(t.polygonOffset(ie,oe),Q=ie,G=oe)):ze(t.POLYGON_OFFSET_FILL)}function $(P){P?Me(t.SCISSOR_TEST):ze(t.SCISSOR_TEST)}function F(P){P===void 0&&(P=t.TEXTURE0+q-1),ee!==P&&(t.activeTexture(P),ee=P)}function K(P,ie,oe){oe===void 0&&(ee===null?oe=t.TEXTURE0+q-1:oe=ee);let ve=k[oe];ve===void 0&&(ve={type:void 0,texture:void 0},k[oe]=ve),(ve.type!==P||ve.texture!==ie)&&(ee!==oe&&(t.activeTexture(oe),ee=oe),t.bindTexture(P,ie||Le[P]),ve.type=P,ve.texture=ie)}function ge(){const P=k[ee];P!==void 0&&P.type!==void 0&&(t.bindTexture(P.type,null),P.type=void 0,P.texture=void 0)}function ae(){try{t.compressedTexImage2D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function le(){try{t.compressedTexImage3D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function fe(){try{t.texSubImage2D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function Be(){try{t.texSubImage3D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function V(){try{t.compressedTexSubImage2D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function tt(){try{t.compressedTexSubImage3D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function we(){try{t.texStorage2D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function ye(){try{t.texStorage3D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function ce(){try{t.texImage2D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function ue(){try{t.texImage3D.apply(t,arguments)}catch(P){console.error("THREE.WebGLState:",P)}}function Je(P){se.equals(P)===!1&&(t.scissor(P.x,P.y,P.z,P.w),se.copy(P))}function Ue(P){me.equals(P)===!1&&(t.viewport(P.x,P.y,P.z,P.w),me.copy(P))}function Qe(P,ie){let oe=d.get(ie);oe===void 0&&(oe=new WeakMap,d.set(ie,oe));let ve=oe.get(P);ve===void 0&&(ve=t.getUniformBlockIndex(ie,P.name),oe.set(P,ve))}function Xe(P,ie){const ve=d.get(ie).get(P);c.get(ie)!==ve&&(t.uniformBlockBinding(ie,ve,P.__bindingPointIndex),c.set(ie,ve))}function ne(){t.disable(t.BLEND),t.disable(t.CULL_FACE),t.disable(t.DEPTH_TEST),t.disable(t.POLYGON_OFFSET_FILL),t.disable(t.SCISSOR_TEST),t.disable(t.STENCIL_TEST),t.disable(t.SAMPLE_ALPHA_TO_COVERAGE),t.blendEquation(t.FUNC_ADD),t.blendFunc(t.ONE,t.ZERO),t.blendFuncSeparate(t.ONE,t.ZERO,t.ONE,t.ZERO),t.blendColor(0,0,0,0),t.colorMask(!0,!0,!0,!0),t.clearColor(0,0,0,0),t.depthMask(!0),t.depthFunc(t.LESS),t.clearDepth(1),t.stencilMask(4294967295),t.stencilFunc(t.ALWAYS,0,4294967295),t.stencilOp(t.KEEP,t.KEEP,t.KEEP),t.clearStencil(0),t.cullFace(t.BACK),t.frontFace(t.CCW),t.polygonOffset(0,0),t.activeTexture(t.TEXTURE0),t.bindFramebuffer(t.FRAMEBUFFER,null),i===!0&&(t.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),t.bindFramebuffer(t.READ_FRAMEBUFFER,null)),t.useProgram(null),t.lineWidth(1),t.scissor(0,0,t.canvas.width,t.canvas.height),t.viewport(0,0,t.canvas.width,t.canvas.height),f={},ee=null,k={},p={},E=new WeakMap,g=[],h=null,u=!1,v=null,y=null,T=null,_=null,C=null,B=null,z=null,M=new Ne(0,0,0),I=0,H=!1,W=null,Y=null,b=null,Q=null,G=null,se.set(0,0,t.canvas.width,t.canvas.height),me.set(0,0,t.canvas.width,t.canvas.height),a.reset(),A.reset(),l.reset()}return{buffers:{color:a,depth:A,stencil:l},enable:Me,disable:ze,bindFramebuffer:L,drawBuffers:St,useProgram:He,setBlending:be,setMaterial:De,setFlipSided:Re,setCullFace:x,setLineWidth:m,setPolygonOffset:O,setScissorTest:$,activeTexture:F,bindTexture:K,unbindTexture:ge,compressedTexImage2D:ae,compressedTexImage3D:le,texImage2D:ce,texImage3D:ue,updateUBOMapping:Qe,uniformBlockBinding:Xe,texStorage2D:we,texStorage3D:ye,texSubImage2D:fe,texSubImage3D:Be,compressedTexSubImage2D:V,compressedTexSubImage3D:tt,scissor:Je,viewport:Ue,reset:ne}}function ih(t,e,n,i,r,s,o){const a=r.isWebGL2,A=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new WeakMap;let d;const f=new WeakMap;let p=!1;try{p=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function E(x,m){return p?new OffscreenCanvas(x,m):Bi("canvas")}function g(x,m,O,$){let F=1;if((x.width>$||x.height>$)&&(F=$/Math.max(x.width,x.height)),F<1||m===!0)if(typeof HTMLImageElement<"u"&&x instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&x instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&x instanceof ImageBitmap){const K=m?bi:Math.floor,ge=K(F*x.width),ae=K(F*x.height);d===void 0&&(d=E(ge,ae));const le=O?E(ge,ae):d;return le.width=ge,le.height=ae,le.getContext("2d").drawImage(x,0,0,ge,ae),console.warn("THREE.WebGLRenderer: Texture has been resized from ("+x.width+"x"+x.height+") to ("+ge+"x"+ae+")."),le}else return"data"in x&&console.warn("THREE.WebGLRenderer: Image in DataTexture is too big ("+x.width+"x"+x.height+")."),x;return x}function h(x){return Br(x.width)&&Br(x.height)}function u(x){return a?!1:x.wrapS!==Bt||x.wrapT!==Bt||x.minFilter!==dt&&x.minFilter!==yt}function v(x,m){return x.generateMipmaps&&m&&x.minFilter!==dt&&x.minFilter!==yt}function y(x){t.generateMipmap(x)}function T(x,m,O,$,F=!1){if(a===!1)return m;if(x!==null){if(t[x]!==void 0)return t[x];console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '"+x+"'")}let K=m;if(m===t.RED&&(O===t.FLOAT&&(K=t.R32F),O===t.HALF_FLOAT&&(K=t.R16F),O===t.UNSIGNED_BYTE&&(K=t.R8)),m===t.RED_INTEGER&&(O===t.UNSIGNED_BYTE&&(K=t.R8UI),O===t.UNSIGNED_SHORT&&(K=t.R16UI),O===t.UNSIGNED_INT&&(K=t.R32UI),O===t.BYTE&&(K=t.R8I),O===t.SHORT&&(K=t.R16I),O===t.INT&&(K=t.R32I)),m===t.RG&&(O===t.FLOAT&&(K=t.RG32F),O===t.HALF_FLOAT&&(K=t.RG16F),O===t.UNSIGNED_BYTE&&(K=t.RG8)),m===t.RGBA){const ge=F?yi:je.getTransfer($);O===t.FLOAT&&(K=t.RGBA32F),O===t.HALF_FLOAT&&(K=t.RGBA16F),O===t.UNSIGNED_BYTE&&(K=ge===We?t.SRGB8_ALPHA8:t.RGBA8),O===t.UNSIGNED_SHORT_4_4_4_4&&(K=t.RGBA4),O===t.UNSIGNED_SHORT_5_5_5_1&&(K=t.RGB5_A1)}return(K===t.R16F||K===t.R32F||K===t.RG16F||K===t.RG32F||K===t.RGBA16F||K===t.RGBA32F)&&e.get("EXT_color_buffer_float"),K}function _(x,m,O){return v(x,O)===!0||x.isFramebufferTexture&&x.minFilter!==dt&&x.minFilter!==yt?Math.log2(Math.max(m.width,m.height))+1:x.mipmaps!==void 0&&x.mipmaps.length>0?x.mipmaps.length:x.isCompressedTexture&&Array.isArray(x.image)?m.mipmaps.length:1}function C(x){return x===dt||x===Ms||x===Er?t.NEAREST:t.LINEAR}function B(x){const m=x.target;m.removeEventListener("dispose",B),M(m),m.isVideoTexture&&c.delete(m)}function z(x){const m=x.target;m.removeEventListener("dispose",z),H(m)}function M(x){const m=i.get(x);if(m.__webglInit===void 0)return;const O=x.source,$=f.get(O);if($){const F=$[m.__cacheKey];F.usedTimes--,F.usedTimes===0&&I(x),Object.keys($).length===0&&f.delete(O)}i.remove(x)}function I(x){const m=i.get(x);t.deleteTexture(m.__webglTexture);const O=x.source,$=f.get(O);delete $[m.__cacheKey],o.memory.textures--}function H(x){const m=x.texture,O=i.get(x),$=i.get(m);if($.__webglTexture!==void 0&&(t.deleteTexture($.__webglTexture),o.memory.textures--),x.depthTexture&&x.depthTexture.dispose(),x.isWebGLCubeRenderTarget)for(let F=0;F<6;F++){if(Array.isArray(O.__webglFramebuffer[F]))for(let K=0;K<O.__webglFramebuffer[F].length;K++)t.deleteFramebuffer(O.__webglFramebuffer[F][K]);else t.deleteFramebuffer(O.__webglFramebuffer[F]);O.__webglDepthbuffer&&t.deleteRenderbuffer(O.__webglDepthbuffer[F])}else{if(Array.isArray(O.__webglFramebuffer))for(let F=0;F<O.__webglFramebuffer.length;F++)t.deleteFramebuffer(O.__webglFramebuffer[F]);else t.deleteFramebuffer(O.__webglFramebuffer);if(O.__webglDepthbuffer&&t.deleteRenderbuffer(O.__webglDepthbuffer),O.__webglMultisampledFramebuffer&&t.deleteFramebuffer(O.__webglMultisampledFramebuffer),O.__webglColorRenderbuffer)for(let F=0;F<O.__webglColorRenderbuffer.length;F++)O.__webglColorRenderbuffer[F]&&t.deleteRenderbuffer(O.__webglColorRenderbuffer[F]);O.__webglDepthRenderbuffer&&t.deleteRenderbuffer(O.__webglDepthRenderbuffer)}if(x.isWebGLMultipleRenderTargets)for(let F=0,K=m.length;F<K;F++){const ge=i.get(m[F]);ge.__webglTexture&&(t.deleteTexture(ge.__webglTexture),o.memory.textures--),i.remove(m[F])}i.remove(m),i.remove(x)}let W=0;function Y(){W=0}function b(){const x=W;return x>=r.maxTextures&&console.warn("THREE.WebGLTextures: Trying to use "+x+" texture units while this GPU supports only "+r.maxTextures),W+=1,x}function Q(x){const m=[];return m.push(x.wrapS),m.push(x.wrapT),m.push(x.wrapR||0),m.push(x.magFilter),m.push(x.minFilter),m.push(x.anisotropy),m.push(x.internalFormat),m.push(x.format),m.push(x.type),m.push(x.generateMipmaps),m.push(x.premultiplyAlpha),m.push(x.flipY),m.push(x.unpackAlignment),m.push(x.colorSpace),m.join()}function G(x,m){const O=i.get(x);if(x.isVideoTexture&&De(x),x.isRenderTargetTexture===!1&&x.version>0&&O.__version!==x.version){const $=x.image;if($===null)console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");else if($.complete===!1)console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");else{se(O,x,m);return}}n.bindTexture(t.TEXTURE_2D,O.__webglTexture,t.TEXTURE0+m)}function q(x,m){const O=i.get(x);if(x.version>0&&O.__version!==x.version){se(O,x,m);return}n.bindTexture(t.TEXTURE_2D_ARRAY,O.__webglTexture,t.TEXTURE0+m)}function U(x,m){const O=i.get(x);if(x.version>0&&O.__version!==x.version){se(O,x,m);return}n.bindTexture(t.TEXTURE_3D,O.__webglTexture,t.TEXTURE0+m)}function j(x,m){const O=i.get(x);if(x.version>0&&O.__version!==x.version){me(O,x,m);return}n.bindTexture(t.TEXTURE_CUBE_MAP,O.__webglTexture,t.TEXTURE0+m)}const X={[gr]:t.REPEAT,[Bt]:t.CLAMP_TO_EDGE,[mr]:t.MIRRORED_REPEAT},ee={[dt]:t.NEAREST,[Ms]:t.NEAREST_MIPMAP_NEAREST,[Er]:t.NEAREST_MIPMAP_LINEAR,[yt]:t.LINEAR,[MA]:t.LINEAR_MIPMAP_NEAREST,[ii]:t.LINEAR_MIPMAP_LINEAR},k={[JA]:t.NEVER,[GA]:t.ALWAYS,[OA]:t.LESS,[aa]:t.LEQUAL,[DA]:t.EQUAL,[NA]:t.GEQUAL,[HA]:t.GREATER,[QA]:t.NOTEQUAL};function Z(x,m,O){if(O?(t.texParameteri(x,t.TEXTURE_WRAP_S,X[m.wrapS]),t.texParameteri(x,t.TEXTURE_WRAP_T,X[m.wrapT]),(x===t.TEXTURE_3D||x===t.TEXTURE_2D_ARRAY)&&t.texParameteri(x,t.TEXTURE_WRAP_R,X[m.wrapR]),t.texParameteri(x,t.TEXTURE_MAG_FILTER,ee[m.magFilter]),t.texParameteri(x,t.TEXTURE_MIN_FILTER,ee[m.minFilter])):(t.texParameteri(x,t.TEXTURE_WRAP_S,t.CLAMP_TO_EDGE),t.texParameteri(x,t.TEXTURE_WRAP_T,t.CLAMP_TO_EDGE),(x===t.TEXTURE_3D||x===t.TEXTURE_2D_ARRAY)&&t.texParameteri(x,t.TEXTURE_WRAP_R,t.CLAMP_TO_EDGE),(m.wrapS!==Bt||m.wrapT!==Bt)&&console.warn("THREE.WebGLRenderer: Texture is not power of two. Texture.wrapS and Texture.wrapT should be set to THREE.ClampToEdgeWrapping."),t.texParameteri(x,t.TEXTURE_MAG_FILTER,C(m.magFilter)),t.texParameteri(x,t.TEXTURE_MIN_FILTER,C(m.minFilter)),m.minFilter!==dt&&m.minFilter!==yt&&console.warn("THREE.WebGLRenderer: Texture is not power of two. Texture.minFilter should be set to THREE.NearestFilter or THREE.LinearFilter.")),m.compareFunction&&(t.texParameteri(x,t.TEXTURE_COMPARE_MODE,t.COMPARE_REF_TO_TEXTURE),t.texParameteri(x,t.TEXTURE_COMPARE_FUNC,k[m.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){const $=e.get("EXT_texture_filter_anisotropic");if(m.magFilter===dt||m.minFilter!==Er&&m.minFilter!==ii||m.type===sn&&e.has("OES_texture_float_linear")===!1||a===!1&&m.type===ri&&e.has("OES_texture_half_float_linear")===!1)return;(m.anisotropy>1||i.get(m).__currentAnisotropy)&&(t.texParameterf(x,$.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(m.anisotropy,r.getMaxAnisotropy())),i.get(m).__currentAnisotropy=m.anisotropy)}}function re(x,m){let O=!1;x.__webglInit===void 0&&(x.__webglInit=!0,m.addEventListener("dispose",B));const $=m.source;let F=f.get($);F===void 0&&(F={},f.set($,F));const K=Q(m);if(K!==x.__cacheKey){F[K]===void 0&&(F[K]={texture:t.createTexture(),usedTimes:0},o.memory.textures++,O=!0),F[K].usedTimes++;const ge=F[x.__cacheKey];ge!==void 0&&(F[x.__cacheKey].usedTimes--,ge.usedTimes===0&&I(m)),x.__cacheKey=K,x.__webglTexture=F[K].texture}return O}function se(x,m,O){let $=t.TEXTURE_2D;(m.isDataArrayTexture||m.isCompressedArrayTexture)&&($=t.TEXTURE_2D_ARRAY),m.isData3DTexture&&($=t.TEXTURE_3D);const F=re(x,m),K=m.source;n.bindTexture($,x.__webglTexture,t.TEXTURE0+O);const ge=i.get(K);if(K.version!==ge.__version||F===!0){n.activeTexture(t.TEXTURE0+O);const ae=je.getPrimaries(je.workingColorSpace),le=m.colorSpace===Ct?null:je.getPrimaries(m.colorSpace),fe=m.colorSpace===Ct||ae===le?t.NONE:t.BROWSER_DEFAULT_WEBGL;t.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,m.flipY),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,m.premultiplyAlpha),t.pixelStorei(t.UNPACK_ALIGNMENT,m.unpackAlignment),t.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,fe);const Be=u(m)&&h(m.image)===!1;let V=g(m.image,Be,!1,r.maxTextureSize);V=Re(m,V);const tt=h(V)||a,we=s.convert(m.format,m.colorSpace);let ye=s.convert(m.type),ce=T(m.internalFormat,we,ye,m.colorSpace,m.isVideoTexture);Z($,m,tt);let ue;const Je=m.mipmaps,Ue=a&&m.isVideoTexture!==!0&&ce!==Ls,Qe=ge.__version===void 0||F===!0,Xe=_(m,V,tt);if(m.isDepthTexture)ce=t.DEPTH_COMPONENT,a?m.type===sn?ce=t.DEPTH_COMPONENT32F:m.type===rn?ce=t.DEPTH_COMPONENT24:m.type===hn?ce=t.DEPTH24_STENCIL8:ce=t.DEPTH_COMPONENT16:m.type===sn&&console.error("WebGLRenderer: Floating point depth texture requires WebGL2."),m.format===pn&&ce===t.DEPTH_COMPONENT&&m.type!==Sr&&m.type!==rn&&(console.warn("THREE.WebGLRenderer: Use UnsignedShortType or UnsignedIntType for DepthFormat DepthTexture."),m.type=rn,ye=s.convert(m.type)),m.format===kn&&ce===t.DEPTH_COMPONENT&&(ce=t.DEPTH_STENCIL,m.type!==hn&&(console.warn("THREE.WebGLRenderer: Use UnsignedInt248Type for DepthStencilFormat DepthTexture."),m.type=hn,ye=s.convert(m.type))),Qe&&(Ue?n.texStorage2D(t.TEXTURE_2D,1,ce,V.width,V.height):n.texImage2D(t.TEXTURE_2D,0,ce,V.width,V.height,0,we,ye,null));else if(m.isDataTexture)if(Je.length>0&&tt){Ue&&Qe&&n.texStorage2D(t.TEXTURE_2D,Xe,ce,Je[0].width,Je[0].height);for(let ne=0,P=Je.length;ne<P;ne++)ue=Je[ne],Ue?n.texSubImage2D(t.TEXTURE_2D,ne,0,0,ue.width,ue.height,we,ye,ue.data):n.texImage2D(t.TEXTURE_2D,ne,ce,ue.width,ue.height,0,we,ye,ue.data);m.generateMipmaps=!1}else Ue?(Qe&&n.texStorage2D(t.TEXTURE_2D,Xe,ce,V.width,V.height),n.texSubImage2D(t.TEXTURE_2D,0,0,0,V.width,V.height,we,ye,V.data)):n.texImage2D(t.TEXTURE_2D,0,ce,V.width,V.height,0,we,ye,V.data);else if(m.isCompressedTexture)if(m.isCompressedArrayTexture){Ue&&Qe&&n.texStorage3D(t.TEXTURE_2D_ARRAY,Xe,ce,Je[0].width,Je[0].height,V.depth);for(let ne=0,P=Je.length;ne<P;ne++)ue=Je[ne],m.format!==kt?we!==null?Ue?n.compressedTexSubImage3D(t.TEXTURE_2D_ARRAY,ne,0,0,0,ue.width,ue.height,V.depth,we,ue.data,0,0):n.compressedTexImage3D(t.TEXTURE_2D_ARRAY,ne,ce,ue.width,ue.height,V.depth,0,ue.data,0,0):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):Ue?n.texSubImage3D(t.TEXTURE_2D_ARRAY,ne,0,0,0,ue.width,ue.height,V.depth,we,ye,ue.data):n.texImage3D(t.TEXTURE_2D_ARRAY,ne,ce,ue.width,ue.height,V.depth,0,we,ye,ue.data)}else{Ue&&Qe&&n.texStorage2D(t.TEXTURE_2D,Xe,ce,Je[0].width,Je[0].height);for(let ne=0,P=Je.length;ne<P;ne++)ue=Je[ne],m.format!==kt?we!==null?Ue?n.compressedTexSubImage2D(t.TEXTURE_2D,ne,0,0,ue.width,ue.height,we,ue.data):n.compressedTexImage2D(t.TEXTURE_2D,ne,ce,ue.width,ue.height,0,ue.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):Ue?n.texSubImage2D(t.TEXTURE_2D,ne,0,0,ue.width,ue.height,we,ye,ue.data):n.texImage2D(t.TEXTURE_2D,ne,ce,ue.width,ue.height,0,we,ye,ue.data)}else if(m.isDataArrayTexture)Ue?(Qe&&n.texStorage3D(t.TEXTURE_2D_ARRAY,Xe,ce,V.width,V.height,V.depth),n.texSubImage3D(t.TEXTURE_2D_ARRAY,0,0,0,0,V.width,V.height,V.depth,we,ye,V.data)):n.texImage3D(t.TEXTURE_2D_ARRAY,0,ce,V.width,V.height,V.depth,0,we,ye,V.data);else if(m.isData3DTexture)Ue?(Qe&&n.texStorage3D(t.TEXTURE_3D,Xe,ce,V.width,V.height,V.depth),n.texSubImage3D(t.TEXTURE_3D,0,0,0,0,V.width,V.height,V.depth,we,ye,V.data)):n.texImage3D(t.TEXTURE_3D,0,ce,V.width,V.height,V.depth,0,we,ye,V.data);else if(m.isFramebufferTexture){if(Qe)if(Ue)n.texStorage2D(t.TEXTURE_2D,Xe,ce,V.width,V.height);else{let ne=V.width,P=V.height;for(let ie=0;ie<Xe;ie++)n.texImage2D(t.TEXTURE_2D,ie,ce,ne,P,0,we,ye,null),ne>>=1,P>>=1}}else if(Je.length>0&&tt){Ue&&Qe&&n.texStorage2D(t.TEXTURE_2D,Xe,ce,Je[0].width,Je[0].height);for(let ne=0,P=Je.length;ne<P;ne++)ue=Je[ne],Ue?n.texSubImage2D(t.TEXTURE_2D,ne,0,0,we,ye,ue):n.texImage2D(t.TEXTURE_2D,ne,ce,we,ye,ue);m.generateMipmaps=!1}else Ue?(Qe&&n.texStorage2D(t.TEXTURE_2D,Xe,ce,V.width,V.height),n.texSubImage2D(t.TEXTURE_2D,0,0,0,we,ye,V)):n.texImage2D(t.TEXTURE_2D,0,ce,we,ye,V);v(m,tt)&&y($),ge.__version=K.version,m.onUpdate&&m.onUpdate(m)}x.__version=m.version}function me(x,m,O){if(m.image.length!==6)return;const $=re(x,m),F=m.source;n.bindTexture(t.TEXTURE_CUBE_MAP,x.__webglTexture,t.TEXTURE0+O);const K=i.get(F);if(F.version!==K.__version||$===!0){n.activeTexture(t.TEXTURE0+O);const ge=je.getPrimaries(je.workingColorSpace),ae=m.colorSpace===Ct?null:je.getPrimaries(m.colorSpace),le=m.colorSpace===Ct||ge===ae?t.NONE:t.BROWSER_DEFAULT_WEBGL;t.pixelStorei(t.UNPACK_FLIP_Y_WEBGL,m.flipY),t.pixelStorei(t.UNPACK_PREMULTIPLY_ALPHA_WEBGL,m.premultiplyAlpha),t.pixelStorei(t.UNPACK_ALIGNMENT,m.unpackAlignment),t.pixelStorei(t.UNPACK_COLORSPACE_CONVERSION_WEBGL,le);const fe=m.isCompressedTexture||m.image[0].isCompressedTexture,Be=m.image[0]&&m.image[0].isDataTexture,V=[];for(let ne=0;ne<6;ne++)!fe&&!Be?V[ne]=g(m.image[ne],!1,!0,r.maxCubemapSize):V[ne]=Be?m.image[ne].image:m.image[ne],V[ne]=Re(m,V[ne]);const tt=V[0],we=h(tt)||a,ye=s.convert(m.format,m.colorSpace),ce=s.convert(m.type),ue=T(m.internalFormat,ye,ce,m.colorSpace),Je=a&&m.isVideoTexture!==!0,Ue=K.__version===void 0||$===!0;let Qe=_(m,tt,we);Z(t.TEXTURE_CUBE_MAP,m,we);let Xe;if(fe){Je&&Ue&&n.texStorage2D(t.TEXTURE_CUBE_MAP,Qe,ue,tt.width,tt.height);for(let ne=0;ne<6;ne++){Xe=V[ne].mipmaps;for(let P=0;P<Xe.length;P++){const ie=Xe[P];m.format!==kt?ye!==null?Je?n.compressedTexSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,P,0,0,ie.width,ie.height,ye,ie.data):n.compressedTexImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,P,ue,ie.width,ie.height,0,ie.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):Je?n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,P,0,0,ie.width,ie.height,ye,ce,ie.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,P,ue,ie.width,ie.height,0,ye,ce,ie.data)}}}else{Xe=m.mipmaps,Je&&Ue&&(Xe.length>0&&Qe++,n.texStorage2D(t.TEXTURE_CUBE_MAP,Qe,ue,V[0].width,V[0].height));for(let ne=0;ne<6;ne++)if(Be){Je?n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,0,0,0,V[ne].width,V[ne].height,ye,ce,V[ne].data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,0,ue,V[ne].width,V[ne].height,0,ye,ce,V[ne].data);for(let P=0;P<Xe.length;P++){const oe=Xe[P].image[ne].image;Je?n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,P+1,0,0,oe.width,oe.height,ye,ce,oe.data):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,P+1,ue,oe.width,oe.height,0,ye,ce,oe.data)}}else{Je?n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,0,0,0,ye,ce,V[ne]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,0,ue,ye,ce,V[ne]);for(let P=0;P<Xe.length;P++){const ie=Xe[P];Je?n.texSubImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,P+1,0,0,ye,ce,ie.image[ne]):n.texImage2D(t.TEXTURE_CUBE_MAP_POSITIVE_X+ne,P+1,ue,ye,ce,ie.image[ne])}}}v(m,we)&&y(t.TEXTURE_CUBE_MAP),K.__version=F.version,m.onUpdate&&m.onUpdate(m)}x.__version=m.version}function xe(x,m,O,$,F,K){const ge=s.convert(O.format,O.colorSpace),ae=s.convert(O.type),le=T(O.internalFormat,ge,ae,O.colorSpace);if(!i.get(m).__hasExternalTextures){const Be=Math.max(1,m.width>>K),V=Math.max(1,m.height>>K);F===t.TEXTURE_3D||F===t.TEXTURE_2D_ARRAY?n.texImage3D(F,K,le,Be,V,m.depth,0,ge,ae,null):n.texImage2D(F,K,le,Be,V,0,ge,ae,null)}n.bindFramebuffer(t.FRAMEBUFFER,x),be(m)?A.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,$,F,i.get(O).__webglTexture,0,Se(m)):(F===t.TEXTURE_2D||F>=t.TEXTURE_CUBE_MAP_POSITIVE_X&&F<=t.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&t.framebufferTexture2D(t.FRAMEBUFFER,$,F,i.get(O).__webglTexture,K),n.bindFramebuffer(t.FRAMEBUFFER,null)}function Le(x,m,O){if(t.bindRenderbuffer(t.RENDERBUFFER,x),m.depthBuffer&&!m.stencilBuffer){let $=a===!0?t.DEPTH_COMPONENT24:t.DEPTH_COMPONENT16;if(O||be(m)){const F=m.depthTexture;F&&F.isDepthTexture&&(F.type===sn?$=t.DEPTH_COMPONENT32F:F.type===rn&&($=t.DEPTH_COMPONENT24));const K=Se(m);be(m)?A.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,K,$,m.width,m.height):t.renderbufferStorageMultisample(t.RENDERBUFFER,K,$,m.width,m.height)}else t.renderbufferStorage(t.RENDERBUFFER,$,m.width,m.height);t.framebufferRenderbuffer(t.FRAMEBUFFER,t.DEPTH_ATTACHMENT,t.RENDERBUFFER,x)}else if(m.depthBuffer&&m.stencilBuffer){const $=Se(m);O&&be(m)===!1?t.renderbufferStorageMultisample(t.RENDERBUFFER,$,t.DEPTH24_STENCIL8,m.width,m.height):be(m)?A.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,$,t.DEPTH24_STENCIL8,m.width,m.height):t.renderbufferStorage(t.RENDERBUFFER,t.DEPTH_STENCIL,m.width,m.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.DEPTH_STENCIL_ATTACHMENT,t.RENDERBUFFER,x)}else{const $=m.isWebGLMultipleRenderTargets===!0?m.texture:[m.texture];for(let F=0;F<$.length;F++){const K=$[F],ge=s.convert(K.format,K.colorSpace),ae=s.convert(K.type),le=T(K.internalFormat,ge,ae,K.colorSpace),fe=Se(m);O&&be(m)===!1?t.renderbufferStorageMultisample(t.RENDERBUFFER,fe,le,m.width,m.height):be(m)?A.renderbufferStorageMultisampleEXT(t.RENDERBUFFER,fe,le,m.width,m.height):t.renderbufferStorage(t.RENDERBUFFER,le,m.width,m.height)}}t.bindRenderbuffer(t.RENDERBUFFER,null)}function Me(x,m){if(m&&m.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(n.bindFramebuffer(t.FRAMEBUFFER,x),!(m.depthTexture&&m.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");(!i.get(m.depthTexture).__webglTexture||m.depthTexture.image.width!==m.width||m.depthTexture.image.height!==m.height)&&(m.depthTexture.image.width=m.width,m.depthTexture.image.height=m.height,m.depthTexture.needsUpdate=!0),G(m.depthTexture,0);const $=i.get(m.depthTexture).__webglTexture,F=Se(m);if(m.depthTexture.format===pn)be(m)?A.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,t.DEPTH_ATTACHMENT,t.TEXTURE_2D,$,0,F):t.framebufferTexture2D(t.FRAMEBUFFER,t.DEPTH_ATTACHMENT,t.TEXTURE_2D,$,0);else if(m.depthTexture.format===kn)be(m)?A.framebufferTexture2DMultisampleEXT(t.FRAMEBUFFER,t.DEPTH_STENCIL_ATTACHMENT,t.TEXTURE_2D,$,0,F):t.framebufferTexture2D(t.FRAMEBUFFER,t.DEPTH_STENCIL_ATTACHMENT,t.TEXTURE_2D,$,0);else throw new Error("Unknown depthTexture format")}function ze(x){const m=i.get(x),O=x.isWebGLCubeRenderTarget===!0;if(x.depthTexture&&!m.__autoAllocateDepthBuffer){if(O)throw new Error("target.depthTexture not supported in Cube render targets");Me(m.__webglFramebuffer,x)}else if(O){m.__webglDepthbuffer=[];for(let $=0;$<6;$++)n.bindFramebuffer(t.FRAMEBUFFER,m.__webglFramebuffer[$]),m.__webglDepthbuffer[$]=t.createRenderbuffer(),Le(m.__webglDepthbuffer[$],x,!1)}else n.bindFramebuffer(t.FRAMEBUFFER,m.__webglFramebuffer),m.__webglDepthbuffer=t.createRenderbuffer(),Le(m.__webglDepthbuffer,x,!1);n.bindFramebuffer(t.FRAMEBUFFER,null)}function L(x,m,O){const $=i.get(x);m!==void 0&&xe($.__webglFramebuffer,x,x.texture,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,0),O!==void 0&&ze(x)}function St(x){const m=x.texture,O=i.get(x),$=i.get(m);x.addEventListener("dispose",z),x.isWebGLMultipleRenderTargets!==!0&&($.__webglTexture===void 0&&($.__webglTexture=t.createTexture()),$.__version=m.version,o.memory.textures++);const F=x.isWebGLCubeRenderTarget===!0,K=x.isWebGLMultipleRenderTargets===!0,ge=h(x)||a;if(F){O.__webglFramebuffer=[];for(let ae=0;ae<6;ae++)if(a&&m.mipmaps&&m.mipmaps.length>0){O.__webglFramebuffer[ae]=[];for(let le=0;le<m.mipmaps.length;le++)O.__webglFramebuffer[ae][le]=t.createFramebuffer()}else O.__webglFramebuffer[ae]=t.createFramebuffer()}else{if(a&&m.mipmaps&&m.mipmaps.length>0){O.__webglFramebuffer=[];for(let ae=0;ae<m.mipmaps.length;ae++)O.__webglFramebuffer[ae]=t.createFramebuffer()}else O.__webglFramebuffer=t.createFramebuffer();if(K)if(r.drawBuffers){const ae=x.texture;for(let le=0,fe=ae.length;le<fe;le++){const Be=i.get(ae[le]);Be.__webglTexture===void 0&&(Be.__webglTexture=t.createTexture(),o.memory.textures++)}}else console.warn("THREE.WebGLRenderer: WebGLMultipleRenderTargets can only be used with WebGL2 or WEBGL_draw_buffers extension.");if(a&&x.samples>0&&be(x)===!1){const ae=K?m:[m];O.__webglMultisampledFramebuffer=t.createFramebuffer(),O.__webglColorRenderbuffer=[],n.bindFramebuffer(t.FRAMEBUFFER,O.__webglMultisampledFramebuffer);for(let le=0;le<ae.length;le++){const fe=ae[le];O.__webglColorRenderbuffer[le]=t.createRenderbuffer(),t.bindRenderbuffer(t.RENDERBUFFER,O.__webglColorRenderbuffer[le]);const Be=s.convert(fe.format,fe.colorSpace),V=s.convert(fe.type),tt=T(fe.internalFormat,Be,V,fe.colorSpace,x.isXRRenderTarget===!0),we=Se(x);t.renderbufferStorageMultisample(t.RENDERBUFFER,we,tt,x.width,x.height),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+le,t.RENDERBUFFER,O.__webglColorRenderbuffer[le])}t.bindRenderbuffer(t.RENDERBUFFER,null),x.depthBuffer&&(O.__webglDepthRenderbuffer=t.createRenderbuffer(),Le(O.__webglDepthRenderbuffer,x,!0)),n.bindFramebuffer(t.FRAMEBUFFER,null)}}if(F){n.bindTexture(t.TEXTURE_CUBE_MAP,$.__webglTexture),Z(t.TEXTURE_CUBE_MAP,m,ge);for(let ae=0;ae<6;ae++)if(a&&m.mipmaps&&m.mipmaps.length>0)for(let le=0;le<m.mipmaps.length;le++)xe(O.__webglFramebuffer[ae][le],x,m,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+ae,le);else xe(O.__webglFramebuffer[ae],x,m,t.COLOR_ATTACHMENT0,t.TEXTURE_CUBE_MAP_POSITIVE_X+ae,0);v(m,ge)&&y(t.TEXTURE_CUBE_MAP),n.unbindTexture()}else if(K){const ae=x.texture;for(let le=0,fe=ae.length;le<fe;le++){const Be=ae[le],V=i.get(Be);n.bindTexture(t.TEXTURE_2D,V.__webglTexture),Z(t.TEXTURE_2D,Be,ge),xe(O.__webglFramebuffer,x,Be,t.COLOR_ATTACHMENT0+le,t.TEXTURE_2D,0),v(Be,ge)&&y(t.TEXTURE_2D)}n.unbindTexture()}else{let ae=t.TEXTURE_2D;if((x.isWebGL3DRenderTarget||x.isWebGLArrayRenderTarget)&&(a?ae=x.isWebGL3DRenderTarget?t.TEXTURE_3D:t.TEXTURE_2D_ARRAY:console.error("THREE.WebGLTextures: THREE.Data3DTexture and THREE.DataArrayTexture only supported with WebGL2.")),n.bindTexture(ae,$.__webglTexture),Z(ae,m,ge),a&&m.mipmaps&&m.mipmaps.length>0)for(let le=0;le<m.mipmaps.length;le++)xe(O.__webglFramebuffer[le],x,m,t.COLOR_ATTACHMENT0,ae,le);else xe(O.__webglFramebuffer,x,m,t.COLOR_ATTACHMENT0,ae,0);v(m,ge)&&y(ae),n.unbindTexture()}x.depthBuffer&&ze(x)}function He(x){const m=h(x)||a,O=x.isWebGLMultipleRenderTargets===!0?x.texture:[x.texture];for(let $=0,F=O.length;$<F;$++){const K=O[$];if(v(K,m)){const ge=x.isWebGLCubeRenderTarget?t.TEXTURE_CUBE_MAP:t.TEXTURE_2D,ae=i.get(K).__webglTexture;n.bindTexture(ge,ae),y(ge),n.unbindTexture()}}}function he(x){if(a&&x.samples>0&&be(x)===!1){const m=x.isWebGLMultipleRenderTargets?x.texture:[x.texture],O=x.width,$=x.height;let F=t.COLOR_BUFFER_BIT;const K=[],ge=x.stencilBuffer?t.DEPTH_STENCIL_ATTACHMENT:t.DEPTH_ATTACHMENT,ae=i.get(x),le=x.isWebGLMultipleRenderTargets===!0;if(le)for(let fe=0;fe<m.length;fe++)n.bindFramebuffer(t.FRAMEBUFFER,ae.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+fe,t.RENDERBUFFER,null),n.bindFramebuffer(t.FRAMEBUFFER,ae.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+fe,t.TEXTURE_2D,null,0);n.bindFramebuffer(t.READ_FRAMEBUFFER,ae.__webglMultisampledFramebuffer),n.bindFramebuffer(t.DRAW_FRAMEBUFFER,ae.__webglFramebuffer);for(let fe=0;fe<m.length;fe++){K.push(t.COLOR_ATTACHMENT0+fe),x.depthBuffer&&K.push(ge);const Be=ae.__ignoreDepthValues!==void 0?ae.__ignoreDepthValues:!1;if(Be===!1&&(x.depthBuffer&&(F|=t.DEPTH_BUFFER_BIT),x.stencilBuffer&&(F|=t.STENCIL_BUFFER_BIT)),le&&t.framebufferRenderbuffer(t.READ_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.RENDERBUFFER,ae.__webglColorRenderbuffer[fe]),Be===!0&&(t.invalidateFramebuffer(t.READ_FRAMEBUFFER,[ge]),t.invalidateFramebuffer(t.DRAW_FRAMEBUFFER,[ge])),le){const V=i.get(m[fe]).__webglTexture;t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0,t.TEXTURE_2D,V,0)}t.blitFramebuffer(0,0,O,$,0,0,O,$,F,t.NEAREST),l&&t.invalidateFramebuffer(t.READ_FRAMEBUFFER,K)}if(n.bindFramebuffer(t.READ_FRAMEBUFFER,null),n.bindFramebuffer(t.DRAW_FRAMEBUFFER,null),le)for(let fe=0;fe<m.length;fe++){n.bindFramebuffer(t.FRAMEBUFFER,ae.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(t.FRAMEBUFFER,t.COLOR_ATTACHMENT0+fe,t.RENDERBUFFER,ae.__webglColorRenderbuffer[fe]);const Be=i.get(m[fe]).__webglTexture;n.bindFramebuffer(t.FRAMEBUFFER,ae.__webglFramebuffer),t.framebufferTexture2D(t.DRAW_FRAMEBUFFER,t.COLOR_ATTACHMENT0+fe,t.TEXTURE_2D,Be,0)}n.bindFramebuffer(t.DRAW_FRAMEBUFFER,ae.__webglMultisampledFramebuffer)}}function Se(x){return Math.min(r.maxSamples,x.samples)}function be(x){const m=i.get(x);return a&&x.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&m.__useRenderToTexture!==!1}function De(x){const m=o.render.frame;c.get(x)!==m&&(c.set(x,m),x.update())}function Re(x,m){const O=x.colorSpace,$=x.format,F=x.type;return x.isCompressedTexture===!0||x.isVideoTexture===!0||x.format===Pr||O!==zt&&O!==Ct&&(je.getTransfer(O)===We?a===!1?e.has("EXT_sRGB")===!0&&$===kt?(x.format=Pr,x.minFilter=yt,x.generateMipmaps=!1):m=ha.sRGBToLinear(m):($!==kt||F!==nn)&&console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):console.error("THREE.WebGLTextures: Unsupported texture color space:",O)),m}this.allocateTextureUnit=b,this.resetTextureUnits=Y,this.setTexture2D=G,this.setTexture2DArray=q,this.setTexture3D=U,this.setTextureCube=j,this.rebindTextures=L,this.setupRenderTarget=St,this.updateRenderTargetMipmap=He,this.updateMultisampleRenderTarget=he,this.setupDepthRenderbuffer=ze,this.setupFrameBufferTexture=xe,this.useMultisampledRTT=be}function rh(t,e,n){const i=n.isWebGL2;function r(s,o=Ct){let a;const A=je.getTransfer(o);if(s===nn)return t.UNSIGNED_BYTE;if(s===ys)return t.UNSIGNED_SHORT_4_4_4_4;if(s===Cs)return t.UNSIGNED_SHORT_5_5_5_1;if(s===IA)return t.BYTE;if(s===yA)return t.SHORT;if(s===Sr)return t.UNSIGNED_SHORT;if(s===Is)return t.INT;if(s===rn)return t.UNSIGNED_INT;if(s===sn)return t.FLOAT;if(s===ri)return i?t.HALF_FLOAT:(a=e.get("OES_texture_half_float"),a!==null?a.HALF_FLOAT_OES:null);if(s===CA)return t.ALPHA;if(s===kt)return t.RGBA;if(s===TA)return t.LUMINANCE;if(s===PA)return t.LUMINANCE_ALPHA;if(s===pn)return t.DEPTH_COMPONENT;if(s===kn)return t.DEPTH_STENCIL;if(s===Pr)return a=e.get("EXT_sRGB"),a!==null?a.SRGB_ALPHA_EXT:null;if(s===bA)return t.RED;if(s===Ts)return t.RED_INTEGER;if(s===BA)return t.RG;if(s===Ps)return t.RG_INTEGER;if(s===bs)return t.RGBA_INTEGER;if(s===vr||s===xr||s===Mr||s===Ir)if(A===We)if(a=e.get("WEBGL_compressed_texture_s3tc_srgb"),a!==null){if(s===vr)return a.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(s===xr)return a.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(s===Mr)return a.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(s===Ir)return a.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(a=e.get("WEBGL_compressed_texture_s3tc"),a!==null){if(s===vr)return a.COMPRESSED_RGB_S3TC_DXT1_EXT;if(s===xr)return a.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(s===Mr)return a.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(s===Ir)return a.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(s===Bs||s===ks||s===Rs||s===_s)if(a=e.get("WEBGL_compressed_texture_pvrtc"),a!==null){if(s===Bs)return a.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(s===ks)return a.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(s===Rs)return a.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(s===_s)return a.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(s===Ls)return a=e.get("WEBGL_compressed_texture_etc1"),a!==null?a.COMPRESSED_RGB_ETC1_WEBGL:null;if(s===ws||s===Js)if(a=e.get("WEBGL_compressed_texture_etc"),a!==null){if(s===ws)return A===We?a.COMPRESSED_SRGB8_ETC2:a.COMPRESSED_RGB8_ETC2;if(s===Js)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:a.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(s===Os||s===Ds||s===Hs||s===Qs||s===Ns||s===Gs||s===zs||s===js||s===Us||s===Xs||s===Zs||s===Fs||s===Ws||s===Ys)if(a=e.get("WEBGL_compressed_texture_astc"),a!==null){if(s===Os)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:a.COMPRESSED_RGBA_ASTC_4x4_KHR;if(s===Ds)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:a.COMPRESSED_RGBA_ASTC_5x4_KHR;if(s===Hs)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:a.COMPRESSED_RGBA_ASTC_5x5_KHR;if(s===Qs)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:a.COMPRESSED_RGBA_ASTC_6x5_KHR;if(s===Ns)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:a.COMPRESSED_RGBA_ASTC_6x6_KHR;if(s===Gs)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:a.COMPRESSED_RGBA_ASTC_8x5_KHR;if(s===zs)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:a.COMPRESSED_RGBA_ASTC_8x6_KHR;if(s===js)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:a.COMPRESSED_RGBA_ASTC_8x8_KHR;if(s===Us)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:a.COMPRESSED_RGBA_ASTC_10x5_KHR;if(s===Xs)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:a.COMPRESSED_RGBA_ASTC_10x6_KHR;if(s===Zs)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:a.COMPRESSED_RGBA_ASTC_10x8_KHR;if(s===Fs)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:a.COMPRESSED_RGBA_ASTC_10x10_KHR;if(s===Ws)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:a.COMPRESSED_RGBA_ASTC_12x10_KHR;if(s===Ys)return A===We?a.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:a.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(s===yr||s===Vs||s===qs)if(a=e.get("EXT_texture_compression_bptc"),a!==null){if(s===yr)return A===We?a.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:a.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(s===Vs)return a.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(s===qs)return a.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(s===kA||s===Ks||s===$s||s===ea)if(a=e.get("EXT_texture_compression_rgtc"),a!==null){if(s===yr)return a.COMPRESSED_RED_RGTC1_EXT;if(s===Ks)return a.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(s===$s)return a.COMPRESSED_RED_GREEN_RGTC2_EXT;if(s===ea)return a.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return s===hn?i?t.UNSIGNED_INT_24_8:(a=e.get("WEBGL_depth_texture"),a!==null?a.UNSIGNED_INT_24_8_WEBGL:null):t[s]!==void 0?t[s]:null}return{convert:r}}var sh=class extends Pt{constructor(t=[]){super(),this.isArrayCamera=!0,this.cameras=t}},pi=class extends Jt{constructor(){super(),this.isGroup=!0,this.type="Group"}},ah={type:"move"},ns=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new pi,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new pi,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new J,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new J),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new pi,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new J,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new J),this._grip}dispatchEvent(t){return this._targetRay!==null&&this._targetRay.dispatchEvent(t),this._grip!==null&&this._grip.dispatchEvent(t),this._hand!==null&&this._hand.dispatchEvent(t),this}connect(t){if(t&&t.hand){const e=this._hand;if(e)for(const n of t.hand.values())this._getHandJoint(e,n)}return this.dispatchEvent({type:"connected",data:t}),this}disconnect(t){return this.dispatchEvent({type:"disconnected",data:t}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(t,e,n){let i=null,r=null,s=null;const o=this._targetRay,a=this._grip,A=this._hand;if(t&&e.session.visibilityState!=="visible-blurred"){if(A&&t.hand){s=!0;for(const E of t.hand.values()){const g=e.getJointPose(E,n),h=this._getHandJoint(A,E);g!==null&&(h.matrix.fromArray(g.transform.matrix),h.matrix.decompose(h.position,h.rotation,h.scale),h.matrixWorldNeedsUpdate=!0,h.jointRadius=g.radius),h.visible=g!==null}const l=A.joints["index-finger-tip"],c=A.joints["thumb-tip"],d=l.position.distanceTo(c.position),f=.02,p=.005;A.inputState.pinching&&d>f+p?(A.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:t.handedness,target:this})):!A.inputState.pinching&&d<=f-p&&(A.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:t.handedness,target:this}))}else a!==null&&t.gripSpace&&(r=e.getPose(t.gripSpace,n),r!==null&&(a.matrix.fromArray(r.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,r.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(r.linearVelocity)):a.hasLinearVelocity=!1,r.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(r.angularVelocity)):a.hasAngularVelocity=!1));o!==null&&(i=e.getPose(t.targetRaySpace,n),i===null&&r!==null&&(i=r),i!==null&&(o.matrix.fromArray(i.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,i.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(i.linearVelocity)):o.hasLinearVelocity=!1,i.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(i.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(ah)))}return o!==null&&(o.visible=i!==null),a!==null&&(a.visible=r!==null),A!==null&&(A.visible=s!==null),this}_getHandJoint(t,e){if(t.joints[e.jointName]===void 0){const n=new pi;n.matrixAutoUpdate=!1,n.visible=!1,t.joints[e.jointName]=n,t.add(n)}return t.joints[e.jointName]}},oh=class extends _n{constructor(t,e){super();const n=this;let i=null,r=1,s=null,o="local-floor",a=1,A=null,l=null,c=null,d=null,f=null,p=null;const E=e.getContextAttributes();let g=null,h=null;const u=[],v=[],y=new Fe;let T=null;const _=new Pt;_.layers.enable(1),_.viewport=new pt;const C=new Pt;C.layers.enable(2),C.viewport=new pt;const B=[_,C],z=new sh;z.layers.enable(1),z.layers.enable(2);let M=null,I=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(k){let Z=u[k];return Z===void 0&&(Z=new ns,u[k]=Z),Z.getTargetRaySpace()},this.getControllerGrip=function(k){let Z=u[k];return Z===void 0&&(Z=new ns,u[k]=Z),Z.getGripSpace()},this.getHand=function(k){let Z=u[k];return Z===void 0&&(Z=new ns,u[k]=Z),Z.getHandSpace()};function H(k){const Z=v.indexOf(k.inputSource);if(Z===-1)return;const re=u[Z];re!==void 0&&(re.update(k.inputSource,k.frame,A||s),re.dispatchEvent({type:k.type,data:k.inputSource}))}function W(){i.removeEventListener("select",H),i.removeEventListener("selectstart",H),i.removeEventListener("selectend",H),i.removeEventListener("squeeze",H),i.removeEventListener("squeezestart",H),i.removeEventListener("squeezeend",H),i.removeEventListener("end",W),i.removeEventListener("inputsourceschange",Y);for(let k=0;k<u.length;k++){const Z=v[k];Z!==null&&(v[k]=null,u[k].disconnect(Z))}M=null,I=null,t.setRenderTarget(g),f=null,d=null,c=null,i=null,h=null,ee.stop(),n.isPresenting=!1,t.setPixelRatio(T),t.setSize(y.width,y.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(k){r=k,n.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(k){o=k,n.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return A||s},this.setReferenceSpace=function(k){A=k},this.getBaseLayer=function(){return d!==null?d:f},this.getBinding=function(){return c},this.getFrame=function(){return p},this.getSession=function(){return i},this.setSession=async function(k){if(i=k,i!==null){if(g=t.getRenderTarget(),i.addEventListener("select",H),i.addEventListener("selectstart",H),i.addEventListener("selectend",H),i.addEventListener("squeeze",H),i.addEventListener("squeezestart",H),i.addEventListener("squeezeend",H),i.addEventListener("end",W),i.addEventListener("inputsourceschange",Y),E.xrCompatible!==!0&&await e.makeXRCompatible(),T=t.getPixelRatio(),t.getSize(y),i.renderState.layers===void 0||t.capabilities.isWebGL2===!1){const Z={antialias:i.renderState.layers===void 0?E.antialias:!0,alpha:!0,depth:E.depth,stencil:E.stencil,framebufferScaleFactor:r};f=new XRWebGLLayer(i,e,Z),i.updateRenderState({baseLayer:f}),t.setPixelRatio(1),t.setSize(f.framebufferWidth,f.framebufferHeight,!1),h=new mn(f.framebufferWidth,f.framebufferHeight,{format:kt,type:nn,colorSpace:t.outputColorSpace,stencilBuffer:E.stencil})}else{let Z=null,re=null,se=null;E.depth&&(se=E.stencil?e.DEPTH24_STENCIL8:e.DEPTH_COMPONENT24,Z=E.stencil?kn:pn,re=E.stencil?hn:rn);const me={colorFormat:e.RGBA8,depthFormat:se,scaleFactor:r};c=new XRWebGLBinding(i,e),d=c.createProjectionLayer(me),i.updateRenderState({layers:[d]}),t.setPixelRatio(1),t.setSize(d.textureWidth,d.textureHeight,!1),h=new mn(d.textureWidth,d.textureHeight,{format:kt,type:nn,depthTexture:new Ya(d.textureWidth,d.textureHeight,re,void 0,void 0,void 0,void 0,void 0,void 0,Z),stencilBuffer:E.stencil,colorSpace:t.outputColorSpace,samples:E.antialias?4:0});const xe=t.properties.get(h);xe.__ignoreDepthValues=d.ignoreDepthValues}h.isXRRenderTarget=!0,this.setFoveation(a),A=null,s=await i.requestReferenceSpace(o),ee.setContext(i),ee.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(i!==null)return i.environmentBlendMode};function Y(k){for(let Z=0;Z<k.removed.length;Z++){const re=k.removed[Z],se=v.indexOf(re);se>=0&&(v[se]=null,u[se].disconnect(re))}for(let Z=0;Z<k.added.length;Z++){const re=k.added[Z];let se=v.indexOf(re);if(se===-1){for(let xe=0;xe<u.length;xe++)if(xe>=v.length){v.push(re),se=xe;break}else if(v[xe]===null){v[xe]=re,se=xe;break}if(se===-1)break}const me=u[se];me&&me.connect(re)}}const b=new J,Q=new J;function G(k,Z,re){b.setFromMatrixPosition(Z.matrixWorld),Q.setFromMatrixPosition(re.matrixWorld);const se=b.distanceTo(Q),me=Z.projectionMatrix.elements,xe=re.projectionMatrix.elements,Le=me[14]/(me[10]-1),Me=me[14]/(me[10]+1),ze=(me[9]+1)/me[5],L=(me[9]-1)/me[5],St=(me[8]-1)/me[0],He=(xe[8]+1)/xe[0],he=Le*St,Se=Le*He,be=se/(-St+He),De=be*-St;Z.matrixWorld.decompose(k.position,k.quaternion,k.scale),k.translateX(De),k.translateZ(be),k.matrixWorld.compose(k.position,k.quaternion,k.scale),k.matrixWorldInverse.copy(k.matrixWorld).invert();const Re=Le+be,x=Me+be,m=he-De,O=Se+(se-De),$=ze*Me/x*Re,F=L*Me/x*Re;k.projectionMatrix.makePerspective(m,O,$,F,Re,x),k.projectionMatrixInverse.copy(k.projectionMatrix).invert()}function q(k,Z){Z===null?k.matrixWorld.copy(k.matrix):k.matrixWorld.multiplyMatrices(Z.matrixWorld,k.matrix),k.matrixWorldInverse.copy(k.matrixWorld).invert()}this.updateCamera=function(k){if(i===null)return;z.near=C.near=_.near=k.near,z.far=C.far=_.far=k.far,(M!==z.near||I!==z.far)&&(i.updateRenderState({depthNear:z.near,depthFar:z.far}),M=z.near,I=z.far);const Z=k.parent,re=z.cameras;q(z,Z);for(let se=0;se<re.length;se++)q(re[se],Z);re.length===2?G(z,_,C):z.projectionMatrix.copy(_.projectionMatrix),U(k,z,Z)};function U(k,Z,re){re===null?k.matrix.copy(Z.matrixWorld):(k.matrix.copy(re.matrixWorld),k.matrix.invert(),k.matrix.multiply(Z.matrixWorld)),k.matrix.decompose(k.position,k.quaternion,k.scale),k.updateMatrixWorld(!0),k.projectionMatrix.copy(Z.projectionMatrix),k.projectionMatrixInverse.copy(Z.projectionMatrixInverse),k.isPerspectiveCamera&&(k.fov=ai*2*Math.atan(1/k.projectionMatrix.elements[5]),k.zoom=1)}this.getCamera=function(){return z},this.getFoveation=function(){if(!(d===null&&f===null))return a},this.setFoveation=function(k){a=k,d!==null&&(d.fixedFoveation=k),f!==null&&f.fixedFoveation!==void 0&&(f.fixedFoveation=k)};let j=null;function X(k,Z){if(l=Z.getViewerPose(A||s),p=Z,l!==null){const re=l.views;f!==null&&(t.setRenderTargetFramebuffer(h,f.framebuffer),t.setRenderTarget(h));let se=!1;re.length!==z.cameras.length&&(z.cameras.length=0,se=!0);for(let me=0;me<re.length;me++){const xe=re[me];let Le=null;if(f!==null)Le=f.getViewport(xe);else{const ze=c.getViewSubImage(d,xe);Le=ze.viewport,me===0&&(t.setRenderTargetTextures(h,ze.colorTexture,d.ignoreDepthValues?void 0:ze.depthStencilTexture),t.setRenderTarget(h))}let Me=B[me];Me===void 0&&(Me=new Pt,Me.layers.enable(me),Me.viewport=new pt,B[me]=Me),Me.matrix.fromArray(xe.transform.matrix),Me.matrix.decompose(Me.position,Me.quaternion,Me.scale),Me.projectionMatrix.fromArray(xe.projectionMatrix),Me.projectionMatrixInverse.copy(Me.projectionMatrix).invert(),Me.viewport.set(Le.x,Le.y,Le.width,Le.height),me===0&&(z.matrix.copy(Me.matrix),z.matrix.decompose(z.position,z.quaternion,z.scale)),se===!0&&z.cameras.push(Me)}}for(let re=0;re<u.length;re++){const se=v[re],me=u[re];se!==null&&me!==void 0&&me.update(se,Z,A||s)}j&&j(k,Z),Z.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:Z}),p=null}const ee=new Ga;ee.setAnimationLoop(X),this.setAnimationLoop=function(k){j=k},this.dispose=function(){}}};function Ah(t,e){function n(h,u){h.matrixAutoUpdate===!0&&h.updateMatrix(),u.value.copy(h.matrix)}function i(h,u){u.color.getRGB(h.fogColor.value,Da(t)),u.isFog?(h.fogNear.value=u.near,h.fogFar.value=u.far):u.isFogExp2&&(h.fogDensity.value=u.density)}function r(h,u,v,y,T){u.isMeshBasicMaterial||u.isMeshLambertMaterial?s(h,u):u.isMeshToonMaterial?(s(h,u),d(h,u)):u.isMeshPhongMaterial?(s(h,u),c(h,u)):u.isMeshStandardMaterial?(s(h,u),f(h,u),u.isMeshPhysicalMaterial&&p(h,u,T)):u.isMeshMatcapMaterial?(s(h,u),E(h,u)):u.isMeshDepthMaterial?s(h,u):u.isMeshDistanceMaterial?(s(h,u),g(h,u)):u.isMeshNormalMaterial?s(h,u):u.isLineBasicMaterial?(o(h,u),u.isLineDashedMaterial&&a(h,u)):u.isPointsMaterial?A(h,u,v,y):u.isSpriteMaterial?l(h,u):u.isShadowMaterial?(h.color.value.copy(u.color),h.opacity.value=u.opacity):u.isShaderMaterial&&(u.uniformsNeedUpdate=!1)}function s(h,u){h.opacity.value=u.opacity,u.color&&h.diffuse.value.copy(u.color),u.emissive&&h.emissive.value.copy(u.emissive).multiplyScalar(u.emissiveIntensity),u.map&&(h.map.value=u.map,n(u.map,h.mapTransform)),u.alphaMap&&(h.alphaMap.value=u.alphaMap,n(u.alphaMap,h.alphaMapTransform)),u.bumpMap&&(h.bumpMap.value=u.bumpMap,n(u.bumpMap,h.bumpMapTransform),h.bumpScale.value=u.bumpScale,u.side===ut&&(h.bumpScale.value*=-1)),u.normalMap&&(h.normalMap.value=u.normalMap,n(u.normalMap,h.normalMapTransform),h.normalScale.value.copy(u.normalScale),u.side===ut&&h.normalScale.value.negate()),u.displacementMap&&(h.displacementMap.value=u.displacementMap,n(u.displacementMap,h.displacementMapTransform),h.displacementScale.value=u.displacementScale,h.displacementBias.value=u.displacementBias),u.emissiveMap&&(h.emissiveMap.value=u.emissiveMap,n(u.emissiveMap,h.emissiveMapTransform)),u.specularMap&&(h.specularMap.value=u.specularMap,n(u.specularMap,h.specularMapTransform)),u.alphaTest>0&&(h.alphaTest.value=u.alphaTest);const v=e.get(u).envMap;if(v&&(h.envMap.value=v,h.flipEnvMap.value=v.isCubeTexture&&v.isRenderTargetTexture===!1?-1:1,h.reflectivity.value=u.reflectivity,h.ior.value=u.ior,h.refractionRatio.value=u.refractionRatio),u.lightMap){h.lightMap.value=u.lightMap;const y=t._useLegacyLights===!0?Math.PI:1;h.lightMapIntensity.value=u.lightMapIntensity*y,n(u.lightMap,h.lightMapTransform)}u.aoMap&&(h.aoMap.value=u.aoMap,h.aoMapIntensity.value=u.aoMapIntensity,n(u.aoMap,h.aoMapTransform))}function o(h,u){h.diffuse.value.copy(u.color),h.opacity.value=u.opacity,u.map&&(h.map.value=u.map,n(u.map,h.mapTransform))}function a(h,u){h.dashSize.value=u.dashSize,h.totalSize.value=u.dashSize+u.gapSize,h.scale.value=u.scale}function A(h,u,v,y){h.diffuse.value.copy(u.color),h.opacity.value=u.opacity,h.size.value=u.size*v,h.scale.value=y*.5,u.map&&(h.map.value=u.map,n(u.map,h.uvTransform)),u.alphaMap&&(h.alphaMap.value=u.alphaMap,n(u.alphaMap,h.alphaMapTransform)),u.alphaTest>0&&(h.alphaTest.value=u.alphaTest)}function l(h,u){h.diffuse.value.copy(u.color),h.opacity.value=u.opacity,h.rotation.value=u.rotation,u.map&&(h.map.value=u.map,n(u.map,h.mapTransform)),u.alphaMap&&(h.alphaMap.value=u.alphaMap,n(u.alphaMap,h.alphaMapTransform)),u.alphaTest>0&&(h.alphaTest.value=u.alphaTest)}function c(h,u){h.specular.value.copy(u.specular),h.shininess.value=Math.max(u.shininess,1e-4)}function d(h,u){u.gradientMap&&(h.gradientMap.value=u.gradientMap)}function f(h,u){h.metalness.value=u.metalness,u.metalnessMap&&(h.metalnessMap.value=u.metalnessMap,n(u.metalnessMap,h.metalnessMapTransform)),h.roughness.value=u.roughness,u.roughnessMap&&(h.roughnessMap.value=u.roughnessMap,n(u.roughnessMap,h.roughnessMapTransform)),e.get(u).envMap&&(h.envMapIntensity.value=u.envMapIntensity)}function p(h,u,v){h.ior.value=u.ior,u.sheen>0&&(h.sheenColor.value.copy(u.sheenColor).multiplyScalar(u.sheen),h.sheenRoughness.value=u.sheenRoughness,u.sheenColorMap&&(h.sheenColorMap.value=u.sheenColorMap,n(u.sheenColorMap,h.sheenColorMapTransform)),u.sheenRoughnessMap&&(h.sheenRoughnessMap.value=u.sheenRoughnessMap,n(u.sheenRoughnessMap,h.sheenRoughnessMapTransform))),u.clearcoat>0&&(h.clearcoat.value=u.clearcoat,h.clearcoatRoughness.value=u.clearcoatRoughness,u.clearcoatMap&&(h.clearcoatMap.value=u.clearcoatMap,n(u.clearcoatMap,h.clearcoatMapTransform)),u.clearcoatRoughnessMap&&(h.clearcoatRoughnessMap.value=u.clearcoatRoughnessMap,n(u.clearcoatRoughnessMap,h.clearcoatRoughnessMapTransform)),u.clearcoatNormalMap&&(h.clearcoatNormalMap.value=u.clearcoatNormalMap,n(u.clearcoatNormalMap,h.clearcoatNormalMapTransform),h.clearcoatNormalScale.value.copy(u.clearcoatNormalScale),u.side===ut&&h.clearcoatNormalScale.value.negate())),u.iridescence>0&&(h.iridescence.value=u.iridescence,h.iridescenceIOR.value=u.iridescenceIOR,h.iridescenceThicknessMinimum.value=u.iridescenceThicknessRange[0],h.iridescenceThicknessMaximum.value=u.iridescenceThicknessRange[1],u.iridescenceMap&&(h.iridescenceMap.value=u.iridescenceMap,n(u.iridescenceMap,h.iridescenceMapTransform)),u.iridescenceThicknessMap&&(h.iridescenceThicknessMap.value=u.iridescenceThicknessMap,n(u.iridescenceThicknessMap,h.iridescenceThicknessMapTransform))),u.transmission>0&&(h.transmission.value=u.transmission,h.transmissionSamplerMap.value=v.texture,h.transmissionSamplerSize.value.set(v.width,v.height),u.transmissionMap&&(h.transmissionMap.value=u.transmissionMap,n(u.transmissionMap,h.transmissionMapTransform)),h.thickness.value=u.thickness,u.thicknessMap&&(h.thicknessMap.value=u.thicknessMap,n(u.thicknessMap,h.thicknessMapTransform)),h.attenuationDistance.value=u.attenuationDistance,h.attenuationColor.value.copy(u.attenuationColor)),u.anisotropy>0&&(h.anisotropyVector.value.set(u.anisotropy*Math.cos(u.anisotropyRotation),u.anisotropy*Math.sin(u.anisotropyRotation)),u.anisotropyMap&&(h.anisotropyMap.value=u.anisotropyMap,n(u.anisotropyMap,h.anisotropyMapTransform))),h.specularIntensity.value=u.specularIntensity,h.specularColor.value.copy(u.specularColor),u.specularColorMap&&(h.specularColorMap.value=u.specularColorMap,n(u.specularColorMap,h.specularColorMapTransform)),u.specularIntensityMap&&(h.specularIntensityMap.value=u.specularIntensityMap,n(u.specularIntensityMap,h.specularIntensityMapTransform))}function E(h,u){u.matcap&&(h.matcap.value=u.matcap)}function g(h,u){const v=e.get(u).light;h.referencePosition.value.setFromMatrixPosition(v.matrixWorld),h.nearDistance.value=v.shadow.camera.near,h.farDistance.value=v.shadow.camera.far}return{refreshFogUniforms:i,refreshMaterialUniforms:r}}function lh(t,e,n,i){let r={},s={},o=[];const a=n.isWebGL2?t.getParameter(t.MAX_UNIFORM_BUFFER_BINDINGS):0;function A(v,y){const T=y.program;i.uniformBlockBinding(v,T)}function l(v,y){let T=r[v.id];T===void 0&&(E(v),T=c(v),r[v.id]=T,v.addEventListener("dispose",h));const _=y.program;i.updateUBOMapping(v,_);const C=e.render.frame;s[v.id]!==C&&(f(v),s[v.id]=C)}function c(v){const y=d();v.__bindingPointIndex=y;const T=t.createBuffer(),_=v.__size,C=v.usage;return t.bindBuffer(t.UNIFORM_BUFFER,T),t.bufferData(t.UNIFORM_BUFFER,_,C),t.bindBuffer(t.UNIFORM_BUFFER,null),t.bindBufferBase(t.UNIFORM_BUFFER,y,T),T}function d(){for(let v=0;v<a;v++)if(o.indexOf(v)===-1)return o.push(v),v;return console.error("THREE.WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function f(v){const y=r[v.id],T=v.uniforms,_=v.__cache;t.bindBuffer(t.UNIFORM_BUFFER,y);for(let C=0,B=T.length;C<B;C++){const z=Array.isArray(T[C])?T[C]:[T[C]];for(let M=0,I=z.length;M<I;M++){const H=z[M];if(p(H,C,M,_)===!0){const W=H.__offset,Y=Array.isArray(H.value)?H.value:[H.value];let b=0;for(let Q=0;Q<Y.length;Q++){const G=Y[Q],q=g(G);typeof G=="number"||typeof G=="boolean"?(H.__data[0]=G,t.bufferSubData(t.UNIFORM_BUFFER,W+b,H.__data)):G.isMatrix3?(H.__data[0]=G.elements[0],H.__data[1]=G.elements[1],H.__data[2]=G.elements[2],H.__data[3]=0,H.__data[4]=G.elements[3],H.__data[5]=G.elements[4],H.__data[6]=G.elements[5],H.__data[7]=0,H.__data[8]=G.elements[6],H.__data[9]=G.elements[7],H.__data[10]=G.elements[8],H.__data[11]=0):(G.toArray(H.__data,b),b+=q.storage/Float32Array.BYTES_PER_ELEMENT)}t.bufferSubData(t.UNIFORM_BUFFER,W,H.__data)}}}t.bindBuffer(t.UNIFORM_BUFFER,null)}function p(v,y,T,_){const C=v.value,B=y+"_"+T;if(_[B]===void 0)return typeof C=="number"||typeof C=="boolean"?_[B]=C:_[B]=C.clone(),!0;{const z=_[B];if(typeof C=="number"||typeof C=="boolean"){if(z!==C)return _[B]=C,!0}else if(z.equals(C)===!1)return z.copy(C),!0}return!1}function E(v){const y=v.uniforms;let T=0;const _=16;for(let B=0,z=y.length;B<z;B++){const M=Array.isArray(y[B])?y[B]:[y[B]];for(let I=0,H=M.length;I<H;I++){const W=M[I],Y=Array.isArray(W.value)?W.value:[W.value];for(let b=0,Q=Y.length;b<Q;b++){const G=Y[b],q=g(G),U=T%_;U!==0&&_-U<q.boundary&&(T+=_-U),W.__data=new Float32Array(q.storage/Float32Array.BYTES_PER_ELEMENT),W.__offset=T,T+=q.storage}}}const C=T%_;return C>0&&(T+=_-C),v.__size=T,v.__cache={},this}function g(v){const y={boundary:0,storage:0};return typeof v=="number"||typeof v=="boolean"?(y.boundary=4,y.storage=4):v.isVector2?(y.boundary=8,y.storage=8):v.isVector3||v.isColor?(y.boundary=16,y.storage=12):v.isVector4?(y.boundary=16,y.storage=16):v.isMatrix3?(y.boundary=48,y.storage=48):v.isMatrix4?(y.boundary=64,y.storage=64):v.isTexture?console.warn("THREE.WebGLRenderer: Texture samplers can not be part of an uniforms group."):console.warn("THREE.WebGLRenderer: Unsupported uniform value type.",v),y}function h(v){const y=v.target;y.removeEventListener("dispose",h);const T=o.indexOf(y.__bindingPointIndex);o.splice(T,1),t.deleteBuffer(r[y.id]),delete r[y.id],delete s[y.id]}function u(){for(const v in r)t.deleteBuffer(r[v]);o=[],r={},s={}}return{bind:A,update:l,dispose:u}}var mo=class{constructor(t={}){const{canvas:e=nl(),context:n=null,depth:i=!0,stencil:r=!0,alpha:s=!1,antialias:o=!1,premultipliedAlpha:a=!0,preserveDrawingBuffer:A=!1,powerPreference:l="default",failIfMajorPerformanceCaveat:c=!1}=t;this.isWebGLRenderer=!0;let d;n!==null?d=n.getContextAttributes().alpha:d=s;const f=new Uint32Array(4),p=new Int32Array(4);let E=null,g=null;const h=[],u=[];this.domElement=e,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this._outputColorSpace=ot,this._useLegacyLights=!1,this.toneMapping=tn,this.toneMappingExposure=1;const v=this;let y=!1,T=0,_=0,C=null,B=-1,z=null;const M=new pt,I=new pt;let H=null;const W=new Ne(0);let Y=0,b=e.width,Q=e.height,G=1,q=null,U=null;const j=new pt(0,0,b,Q),X=new pt(0,0,b,Q);let ee=!1;const k=new Na;let Z=!1,re=!1,se=null;const me=new gt,xe=new Fe,Le=new J,Me={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};function ze(){return C===null?G:1}let L=n;function St(S,R){for(let D=0;D<S.length;D++){const N=S[D],w=e.getContext(N,R);if(w!==null)return w}return null}try{const S={alpha:!0,depth:i,stencil:r,antialias:o,premultipliedAlpha:a,preserveDrawingBuffer:A,powerPreference:l,failIfMajorPerformanceCaveat:c};if("setAttribute"in e&&e.setAttribute("data-engine",`three.js r${ur}`),e.addEventListener("webglcontextlost",Xe,!1),e.addEventListener("webglcontextrestored",ne,!1),e.addEventListener("webglcontextcreationerror",P,!1),L===null){const R=["webgl2","webgl","experimental-webgl"];if(v.isWebGL1Renderer===!0&&R.shift(),L=St(R,S),L===null)throw St(R)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}typeof WebGLRenderingContext<"u"&&L instanceof WebGLRenderingContext&&console.warn("THREE.WebGLRenderer: WebGL 1 support was deprecated in r153 and will be removed in r163."),L.getShaderPrecisionFormat===void 0&&(L.getShaderPrecisionFormat=function(){return{rangeMin:1,rangeMax:1,precision:1}})}catch(S){throw console.error("THREE.WebGLRenderer: "+S.message),S}let He,he,Se,be,De,Re,x,m,O,$,F,K,ge,ae,le,fe,Be,V,tt,we,ye,ce,ue,Je;function Ue(){He=new Ed(L),he=new ud(L,He,t),He.init(he),ce=new rh(L,He,he),Se=new nh(L,He,he),be=new xd(L),De=new zf,Re=new ih(L,He,Se,De,he,ce,be),x=new fd(v),m=new md(v),O=new Pl(L,he),ue=new ld(L,He,O,he),$=new Sd(L,O,be,ue),F=new Cd(L,$,O,be),tt=new yd(L,he,Re),fe=new dd(De),K=new Gf(v,x,m,He,he,ue,fe),ge=new Ah(v,De),ae=new Uf,le=new Vf(He,he),V=new Ad(v,x,m,Se,F,d,a),Be=new th(v,F,he),Je=new lh(L,be,he,Se),we=new cd(L,He,be,he),ye=new vd(L,He,be,he),be.programs=K.programs,v.capabilities=he,v.extensions=He,v.properties=De,v.renderLists=ae,v.shadowMap=Be,v.state=Se,v.info=be}Ue();const Qe=new oh(v,L);this.xr=Qe,this.getContext=function(){return L},this.getContextAttributes=function(){return L.getContextAttributes()},this.forceContextLoss=function(){const S=He.get("WEBGL_lose_context");S&&S.loseContext()},this.forceContextRestore=function(){const S=He.get("WEBGL_lose_context");S&&S.restoreContext()},this.getPixelRatio=function(){return G},this.setPixelRatio=function(S){S!==void 0&&(G=S,this.setSize(b,Q,!1))},this.getSize=function(S){return S.set(b,Q)},this.setSize=function(S,R,D=!0){if(Qe.isPresenting){console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting.");return}b=S,Q=R,e.width=Math.floor(S*G),e.height=Math.floor(R*G),D===!0&&(e.style.width=S+"px",e.style.height=R+"px"),this.setViewport(0,0,S,R)},this.getDrawingBufferSize=function(S){return S.set(b*G,Q*G).floor()},this.setDrawingBufferSize=function(S,R,D){b=S,Q=R,G=D,e.width=Math.floor(S*D),e.height=Math.floor(R*D),this.setViewport(0,0,S,R)},this.getCurrentViewport=function(S){return S.copy(M)},this.getViewport=function(S){return S.copy(j)},this.setViewport=function(S,R,D,N){S.isVector4?j.set(S.x,S.y,S.z,S.w):j.set(S,R,D,N),Se.viewport(M.copy(j).multiplyScalar(G).floor())},this.getScissor=function(S){return S.copy(X)},this.setScissor=function(S,R,D,N){S.isVector4?X.set(S.x,S.y,S.z,S.w):X.set(S,R,D,N),Se.scissor(I.copy(X).multiplyScalar(G).floor())},this.getScissorTest=function(){return ee},this.setScissorTest=function(S){Se.setScissorTest(ee=S)},this.setOpaqueSort=function(S){q=S},this.setTransparentSort=function(S){U=S},this.getClearColor=function(S){return S.copy(V.getClearColor())},this.setClearColor=function(){V.setClearColor.apply(V,arguments)},this.getClearAlpha=function(){return V.getClearAlpha()},this.setClearAlpha=function(){V.setClearAlpha.apply(V,arguments)},this.clear=function(S=!0,R=!0,D=!0){let N=0;if(S){let w=!1;if(C!==null){const Ae=C.texture.format;w=Ae===bs||Ae===Ps||Ae===Ts}if(w){const Ae=C.texture.type,de=Ae===nn||Ae===rn||Ae===Sr||Ae===hn||Ae===ys||Ae===Cs,Ee=V.getClearColor(),Ie=V.getClearAlpha(),ke=Ee.r,Ce=Ee.g,Te=Ee.b;de?(f[0]=ke,f[1]=Ce,f[2]=Te,f[3]=Ie,L.clearBufferuiv(L.COLOR,0,f)):(p[0]=ke,p[1]=Ce,p[2]=Te,p[3]=Ie,L.clearBufferiv(L.COLOR,0,p))}else N|=L.COLOR_BUFFER_BIT}R&&(N|=L.DEPTH_BUFFER_BIT),D&&(N|=L.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),L.clear(N)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){e.removeEventListener("webglcontextlost",Xe,!1),e.removeEventListener("webglcontextrestored",ne,!1),e.removeEventListener("webglcontextcreationerror",P,!1),ae.dispose(),le.dispose(),De.dispose(),x.dispose(),m.dispose(),F.dispose(),ue.dispose(),Je.dispose(),K.dispose(),Qe.dispose(),Qe.removeEventListener("sessionstart",rt),Qe.removeEventListener("sessionend",vt),se&&(se.dispose(),se=null),Ge.stop()};function Xe(S){S.preventDefault(),console.log("THREE.WebGLRenderer: Context Lost."),y=!0}function ne(){console.log("THREE.WebGLRenderer: Context Restored."),y=!1;const S=be.autoReset,R=Be.enabled,D=Be.autoUpdate,N=Be.needsUpdate,w=Be.type;Ue(),be.autoReset=S,Be.enabled=R,Be.autoUpdate=D,Be.needsUpdate=N,Be.type=w}function P(S){console.error("THREE.WebGLRenderer: A WebGL context could not be created. Reason: ",S.statusMessage)}function ie(S){const R=S.target;R.removeEventListener("dispose",ie),oe(R)}function oe(S){ve(S),De.remove(S)}function ve(S){const R=De.get(S).programs;R!==void 0&&(R.forEach(function(D){K.releaseProgram(D)}),S.isShaderMaterial&&K.releaseShaderCache(S))}this.renderBufferDirect=function(S,R,D,N,w,Ae){R===null&&(R=Me);const de=w.isMesh&&w.matrixWorld.determinant()<0,Ee=Zh(S,R,D,N,w);Se.setMaterial(N,de);let Ie=D.index,ke=1;if(N.wireframe===!0){if(Ie=$.getWireframeAttribute(D),Ie===void 0)return;ke=2}const Ce=D.drawRange,Te=D.attributes.position;let $e=Ce.start*ke,It=(Ce.start+Ce.count)*ke;Ae!==null&&($e=Math.max($e,Ae.start*ke),It=Math.min(It,(Ae.start+Ae.count)*ke)),Ie!==null?($e=Math.max($e,0),It=Math.min(It,Ie.count)):Te!=null&&($e=Math.max($e,0),It=Math.min(It,Te.count));const at=It-$e;if(at<0||at===1/0)return;ue.setup(w,N,Ee,D,Ie);let Kt,Ve=we;if(Ie!==null&&(Kt=O.get(Ie),Ve=ye,Ve.setIndex(Kt)),w.isMesh)N.wireframe===!0?(Se.setLineWidth(N.wireframeLinewidth*ze()),Ve.setMode(L.LINES)):Ve.setMode(L.TRIANGLES);else if(w.isLine){let _e=N.linewidth;_e===void 0&&(_e=1),Se.setLineWidth(_e*ze()),w.isLineSegments?Ve.setMode(L.LINES):w.isLineLoop?Ve.setMode(L.LINE_LOOP):Ve.setMode(L.LINE_STRIP)}else w.isPoints?Ve.setMode(L.POINTS):w.isSprite&&Ve.setMode(L.TRIANGLES);if(w.isBatchedMesh)Ve.renderMultiDraw(w._multiDrawStarts,w._multiDrawCounts,w._multiDrawCount);else if(w.isInstancedMesh)Ve.renderInstances($e,at,w.count);else if(D.isInstancedBufferGeometry){const _e=D._maxInstanceCount!==void 0?D._maxInstanceCount:1/0,os=Math.min(D.instanceCount,_e);Ve.renderInstances($e,at,os)}else Ve.render($e,at)};function pe(S,R,D){S.transparent===!0&&S.side===Gt&&S.forceSinglePass===!1?(S.side=ut,S.needsUpdate=!0,or(S,R,D),S.side=$t,S.needsUpdate=!0,or(S,R,D),S.side=Gt):or(S,R,D)}this.compile=function(S,R,D=null){D===null&&(D=S),g=le.get(D),g.init(),u.push(g),D.traverseVisible(function(w){w.isLight&&w.layers.test(R.layers)&&(g.pushLight(w),w.castShadow&&g.pushShadow(w))}),S!==D&&S.traverseVisible(function(w){w.isLight&&w.layers.test(R.layers)&&(g.pushLight(w),w.castShadow&&g.pushShadow(w))}),g.setupLights(v._useLegacyLights);const N=new Set;return S.traverse(function(w){const Ae=w.material;if(Ae)if(Array.isArray(Ae))for(let de=0;de<Ae.length;de++){const Ee=Ae[de];pe(Ee,D,w),N.add(Ee)}else pe(Ae,D,w),N.add(Ae)}),u.pop(),g=null,N},this.compileAsync=function(S,R,D=null){const N=this.compile(S,R,D);return new Promise(w=>{function Ae(){if(N.forEach(function(de){De.get(de).currentProgram.isReady()&&N.delete(de)}),N.size===0){w(S);return}setTimeout(Ae,10)}He.get("KHR_parallel_shader_compile")!==null?Ae():setTimeout(Ae,10)})};let Ze=null;function Ye(S){Ze&&Ze(S)}function rt(){Ge.stop()}function vt(){Ge.start()}const Ge=new Ga;Ge.setAnimationLoop(Ye),typeof self<"u"&&Ge.setContext(self),this.setAnimationLoop=function(S){Ze=S,Qe.setAnimationLoop(S),S===null?Ge.stop():Ge.start()},Qe.addEventListener("sessionstart",rt),Qe.addEventListener("sessionend",vt),this.render=function(S,R){if(R!==void 0&&R.isCamera!==!0){console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(y===!0)return;S.matrixWorldAutoUpdate===!0&&S.updateMatrixWorld(),R.parent===null&&R.matrixWorldAutoUpdate===!0&&R.updateMatrixWorld(),Qe.enabled===!0&&Qe.isPresenting===!0&&(Qe.cameraAutoUpdate===!0&&Qe.updateCamera(R),R=Qe.getCamera()),S.isScene===!0&&S.onBeforeRender(v,S,R,C),g=le.get(S,u.length),g.init(),u.push(g),me.multiplyMatrices(R.projectionMatrix,R.matrixWorldInverse),k.setFromProjectionMatrix(me),re=this.localClippingEnabled,Z=fe.init(this.clippingPlanes,re),E=ae.get(S,h.length),E.init(),h.push(E),Ht(S,R,0,v.sortObjects),E.finish(),v.sortObjects===!0&&E.sort(q,U),this.info.render.frame++,Z===!0&&fe.beginShadows();const D=g.state.shadowsArray;if(Be.render(D,S,R),Z===!0&&fe.endShadows(),this.info.autoReset===!0&&this.info.reset(),V.render(E,S),g.setupLights(v._useLegacyLights),R.isArrayCamera){const N=R.cameras;for(let w=0,Ae=N.length;w<Ae;w++){const de=N[w];Qt(E,S,de,de.viewport)}}else Qt(E,S,R);C!==null&&(Re.updateMultisampleRenderTarget(C),Re.updateRenderTargetMipmap(C)),S.isScene===!0&&S.onAfterRender(v,S,R),ue.resetDefaultState(),B=-1,z=null,u.pop(),u.length>0?g=u[u.length-1]:g=null,h.pop(),h.length>0?E=h[h.length-1]:E=null};function Ht(S,R,D,N){if(S.visible===!1)return;if(S.layers.test(R.layers)){if(S.isGroup)D=S.renderOrder;else if(S.isLOD)S.autoUpdate===!0&&S.update(R);else if(S.isLight)g.pushLight(S),S.castShadow&&g.pushShadow(S);else if(S.isSprite){if(!S.frustumCulled||k.intersectsSprite(S)){N&&Le.setFromMatrixPosition(S.matrixWorld).applyMatrix4(me);const de=F.update(S),Ee=S.material;Ee.visible&&E.push(S,de,Ee,D,Le.z,null)}}else if((S.isMesh||S.isLine||S.isPoints)&&(!S.frustumCulled||k.intersectsObject(S))){const de=F.update(S),Ee=S.material;if(N&&(S.boundingSphere!==void 0?(S.boundingSphere===null&&S.computeBoundingSphere(),Le.copy(S.boundingSphere.center)):(de.boundingSphere===null&&de.computeBoundingSphere(),Le.copy(de.boundingSphere.center)),Le.applyMatrix4(S.matrixWorld).applyMatrix4(me)),Array.isArray(Ee)){const Ie=de.groups;for(let ke=0,Ce=Ie.length;ke<Ce;ke++){const Te=Ie[ke],$e=Ee[Te.materialIndex];$e&&$e.visible&&E.push(S,de,$e,D,Le.z,Te)}}else Ee.visible&&E.push(S,de,Ee,D,Le.z,null)}}const Ae=S.children;for(let de=0,Ee=Ae.length;de<Ee;de++)Ht(Ae[de],R,D,N)}function Qt(S,R,D,N){const w=S.opaque,Ae=S.transmissive,de=S.transparent;g.setupLightsView(D),Z===!0&&fe.setGlobalState(v.clippingPlanes,D),Ae.length>0&&Xh(w,Ae,R,D),N&&Se.viewport(M.copy(N)),w.length>0&&ar(w,R,D),Ae.length>0&&ar(Ae,R,D),de.length>0&&ar(de,R,D),Se.buffers.depth.setTest(!0),Se.buffers.depth.setMask(!0),Se.buffers.color.setMask(!0),Se.setPolygonOffset(!1)}function Xh(S,R,D,N){if((D.isScene===!0?D.overrideMaterial:null)!==null)return;const Ae=he.isWebGL2;se===null&&(se=new mn(1,1,{generateMipmaps:!0,type:He.has("EXT_color_buffer_half_float")?ri:nn,minFilter:ii,samples:Ae?4:0})),v.getDrawingBufferSize(xe),Ae?se.setSize(xe.x,xe.y):se.setSize(bi(xe.x),bi(xe.y));const de=v.getRenderTarget();v.setRenderTarget(se),v.getClearColor(W),Y=v.getClearAlpha(),Y<1&&v.setClearColor(16777215,.5),v.clear();const Ee=v.toneMapping;v.toneMapping=tn,ar(S,D,N),Re.updateMultisampleRenderTarget(se),Re.updateRenderTargetMipmap(se);let Ie=!1;for(let ke=0,Ce=R.length;ke<Ce;ke++){const Te=R[ke],$e=Te.object,It=Te.geometry,at=Te.material,Kt=Te.group;if(at.side===Gt&&$e.layers.test(N.layers)){const Ve=at.side;at.side=ut,at.needsUpdate=!0,Mo($e,D,N,It,at,Kt),at.side=Ve,at.needsUpdate=!0,Ie=!0}}Ie===!0&&(Re.updateMultisampleRenderTarget(se),Re.updateRenderTargetMipmap(se)),v.setRenderTarget(de),v.setClearColor(W,Y),v.toneMapping=Ee}function ar(S,R,D){const N=R.isScene===!0?R.overrideMaterial:null;for(let w=0,Ae=S.length;w<Ae;w++){const de=S[w],Ee=de.object,Ie=de.geometry,ke=N===null?de.material:N,Ce=de.group;Ee.layers.test(D.layers)&&Mo(Ee,R,D,Ie,ke,Ce)}}function Mo(S,R,D,N,w,Ae){S.onBeforeRender(v,R,D,N,w,Ae),S.modelViewMatrix.multiplyMatrices(D.matrixWorldInverse,S.matrixWorld),S.normalMatrix.getNormalMatrix(S.modelViewMatrix),w.onBeforeRender(v,R,D,N,S,Ae),w.transparent===!0&&w.side===Gt&&w.forceSinglePass===!1?(w.side=ut,w.needsUpdate=!0,v.renderBufferDirect(D,R,N,w,S,Ae),w.side=$t,w.needsUpdate=!0,v.renderBufferDirect(D,R,N,w,S,Ae),w.side=Gt):v.renderBufferDirect(D,R,N,w,S,Ae),S.onAfterRender(v,R,D,N,w,Ae)}function or(S,R,D){R.isScene!==!0&&(R=Me);const N=De.get(S),w=g.state.lights,Ae=g.state.shadowsArray,de=w.state.version,Ee=K.getParameters(S,w.state,Ae,R,D),Ie=K.getProgramCacheKey(Ee);let ke=N.programs;N.environment=S.isMeshStandardMaterial?R.environment:null,N.fog=R.fog,N.envMap=(S.isMeshStandardMaterial?m:x).get(S.envMap||N.environment),ke===void 0&&(S.addEventListener("dispose",ie),ke=new Map,N.programs=ke);let Ce=ke.get(Ie);if(Ce!==void 0){if(N.currentProgram===Ce&&N.lightsStateVersion===de)return yo(S,Ee),Ce}else Ee.uniforms=K.getUniforms(S),S.onBuild(D,Ee,v),S.onBeforeCompile(Ee,v),Ce=K.acquireProgram(Ee,Ie),ke.set(Ie,Ce),N.uniforms=Ee.uniforms;const Te=N.uniforms;return(!S.isShaderMaterial&&!S.isRawShaderMaterial||S.clipping===!0)&&(Te.clippingPlanes=fe.uniform),yo(S,Ee),N.needsLights=Wh(S),N.lightsStateVersion=de,N.needsLights&&(Te.ambientLightColor.value=w.state.ambient,Te.lightProbe.value=w.state.probe,Te.directionalLights.value=w.state.directional,Te.directionalLightShadows.value=w.state.directionalShadow,Te.spotLights.value=w.state.spot,Te.spotLightShadows.value=w.state.spotShadow,Te.rectAreaLights.value=w.state.rectArea,Te.ltc_1.value=w.state.rectAreaLTC1,Te.ltc_2.value=w.state.rectAreaLTC2,Te.pointLights.value=w.state.point,Te.pointLightShadows.value=w.state.pointShadow,Te.hemisphereLights.value=w.state.hemi,Te.directionalShadowMap.value=w.state.directionalShadowMap,Te.directionalShadowMatrix.value=w.state.directionalShadowMatrix,Te.spotShadowMap.value=w.state.spotShadowMap,Te.spotLightMatrix.value=w.state.spotLightMatrix,Te.spotLightMap.value=w.state.spotLightMap,Te.pointShadowMap.value=w.state.pointShadowMap,Te.pointShadowMatrix.value=w.state.pointShadowMatrix),N.currentProgram=Ce,N.uniformsList=null,Ce}function Io(S){if(S.uniformsList===null){const R=S.currentProgram.getUniforms();S.uniformsList=tr.seqWithValue(R.seq,S.uniforms)}return S.uniformsList}function yo(S,R){const D=De.get(S);D.outputColorSpace=R.outputColorSpace,D.batching=R.batching,D.instancing=R.instancing,D.instancingColor=R.instancingColor,D.skinning=R.skinning,D.morphTargets=R.morphTargets,D.morphNormals=R.morphNormals,D.morphColors=R.morphColors,D.morphTargetsCount=R.morphTargetsCount,D.numClippingPlanes=R.numClippingPlanes,D.numIntersection=R.numClipIntersection,D.vertexAlphas=R.vertexAlphas,D.vertexTangents=R.vertexTangents,D.toneMapping=R.toneMapping}function Zh(S,R,D,N,w){R.isScene!==!0&&(R=Me),Re.resetTextureUnits();const Ae=R.fog,de=N.isMeshStandardMaterial?R.environment:null,Ee=C===null?v.outputColorSpace:C.isXRRenderTarget===!0?C.texture.colorSpace:zt,Ie=(N.isMeshStandardMaterial?m:x).get(N.envMap||de),ke=N.vertexColors===!0&&!!D.attributes.color&&D.attributes.color.itemSize===4,Ce=!!D.attributes.tangent&&(!!N.normalMap||N.anisotropy>0),Te=!!D.morphAttributes.position,$e=!!D.morphAttributes.normal,It=!!D.morphAttributes.color;let at=tn;N.toneMapped&&(C===null||C.isXRRenderTarget===!0)&&(at=v.toneMapping);const Kt=D.morphAttributes.position||D.morphAttributes.normal||D.morphAttributes.color,Ve=Kt!==void 0?Kt.length:0,_e=De.get(N),os=g.state.lights;if(Z===!0&&(re===!0||S!==z)){const bt=S===z&&N.id===B;fe.setState(N,S,bt)}let Ke=!1;N.version===_e.__version?(_e.needsLights&&_e.lightsStateVersion!==os.state.version||_e.outputColorSpace!==Ee||w.isBatchedMesh&&_e.batching===!1||!w.isBatchedMesh&&_e.batching===!0||w.isInstancedMesh&&_e.instancing===!1||!w.isInstancedMesh&&_e.instancing===!0||w.isSkinnedMesh&&_e.skinning===!1||!w.isSkinnedMesh&&_e.skinning===!0||w.isInstancedMesh&&_e.instancingColor===!0&&w.instanceColor===null||w.isInstancedMesh&&_e.instancingColor===!1&&w.instanceColor!==null||_e.envMap!==Ie||N.fog===!0&&_e.fog!==Ae||_e.numClippingPlanes!==void 0&&(_e.numClippingPlanes!==fe.numPlanes||_e.numIntersection!==fe.numIntersection)||_e.vertexAlphas!==ke||_e.vertexTangents!==Ce||_e.morphTargets!==Te||_e.morphNormals!==$e||_e.morphColors!==It||_e.toneMapping!==at||he.isWebGL2===!0&&_e.morphTargetsCount!==Ve)&&(Ke=!0):(Ke=!0,_e.__version=N.version);let Cn=_e.currentProgram;Ke===!0&&(Cn=or(N,R,w));let Co=!1,Ei=!1,As=!1;const ct=Cn.getUniforms(),Tn=_e.uniforms;if(Se.useProgram(Cn.program)&&(Co=!0,Ei=!0,As=!0),N.id!==B&&(B=N.id,Ei=!0),Co||z!==S){ct.setValue(L,"projectionMatrix",S.projectionMatrix),ct.setValue(L,"viewMatrix",S.matrixWorldInverse);const bt=ct.map.cameraPosition;bt!==void 0&&bt.setValue(L,Le.setFromMatrixPosition(S.matrixWorld)),he.logarithmicDepthBuffer&&ct.setValue(L,"logDepthBufFC",2/(Math.log(S.far+1)/Math.LN2)),(N.isMeshPhongMaterial||N.isMeshToonMaterial||N.isMeshLambertMaterial||N.isMeshBasicMaterial||N.isMeshStandardMaterial||N.isShaderMaterial)&&ct.setValue(L,"isOrthographic",S.isOrthographicCamera===!0),z!==S&&(z=S,Ei=!0,As=!0)}if(w.isSkinnedMesh){ct.setOptional(L,w,"bindMatrix"),ct.setOptional(L,w,"bindMatrixInverse");const bt=w.skeleton;bt&&(he.floatVertexTextures?(bt.boneTexture===null&&bt.computeBoneTexture(),ct.setValue(L,"boneTexture",bt.boneTexture,Re)):console.warn("THREE.WebGLRenderer: SkinnedMesh can only be used with WebGL 2. With WebGL 1 OES_texture_float and vertex textures support is required."))}w.isBatchedMesh&&(ct.setOptional(L,w,"batchingTexture"),ct.setValue(L,"batchingTexture",w._matricesTexture,Re));const ls=D.morphAttributes;if((ls.position!==void 0||ls.normal!==void 0||ls.color!==void 0&&he.isWebGL2===!0)&&tt.update(w,D,Cn),(Ei||_e.receiveShadow!==w.receiveShadow)&&(_e.receiveShadow=w.receiveShadow,ct.setValue(L,"receiveShadow",w.receiveShadow)),N.isMeshGouraudMaterial&&N.envMap!==null&&(Tn.envMap.value=Ie,Tn.flipEnvMap.value=Ie.isCubeTexture&&Ie.isRenderTargetTexture===!1?-1:1),Ei&&(ct.setValue(L,"toneMappingExposure",v.toneMappingExposure),_e.needsLights&&Fh(Tn,As),Ae&&N.fog===!0&&ge.refreshFogUniforms(Tn,Ae),ge.refreshMaterialUniforms(Tn,N,G,Q,se),tr.upload(L,Io(_e),Tn,Re)),N.isShaderMaterial&&N.uniformsNeedUpdate===!0&&(tr.upload(L,Io(_e),Tn,Re),N.uniformsNeedUpdate=!1),N.isSpriteMaterial&&ct.setValue(L,"center",w.center),ct.setValue(L,"modelViewMatrix",w.modelViewMatrix),ct.setValue(L,"normalMatrix",w.normalMatrix),ct.setValue(L,"modelMatrix",w.matrixWorld),N.isShaderMaterial||N.isRawShaderMaterial){const bt=N.uniformsGroups;for(let cs=0,Yh=bt.length;cs<Yh;cs++)if(he.isWebGL2){const To=bt[cs];Je.update(To,Cn),Je.bind(To,Cn)}else console.warn("THREE.WebGLRenderer: Uniform Buffer Objects can only be used with WebGL 2.")}return Cn}function Fh(S,R){S.ambientLightColor.needsUpdate=R,S.lightProbe.needsUpdate=R,S.directionalLights.needsUpdate=R,S.directionalLightShadows.needsUpdate=R,S.pointLights.needsUpdate=R,S.pointLightShadows.needsUpdate=R,S.spotLights.needsUpdate=R,S.spotLightShadows.needsUpdate=R,S.rectAreaLights.needsUpdate=R,S.hemisphereLights.needsUpdate=R}function Wh(S){return S.isMeshLambertMaterial||S.isMeshToonMaterial||S.isMeshPhongMaterial||S.isMeshStandardMaterial||S.isShadowMaterial||S.isShaderMaterial&&S.lights===!0}this.getActiveCubeFace=function(){return T},this.getActiveMipmapLevel=function(){return _},this.getRenderTarget=function(){return C},this.setRenderTargetTextures=function(S,R,D){De.get(S.texture).__webglTexture=R,De.get(S.depthTexture).__webglTexture=D;const N=De.get(S);N.__hasExternalTextures=!0,N.__hasExternalTextures&&(N.__autoAllocateDepthBuffer=D===void 0,N.__autoAllocateDepthBuffer||He.has("WEBGL_multisampled_render_to_texture")===!0&&(console.warn("THREE.WebGLRenderer: Render-to-texture extension was disabled because an external texture was provided"),N.__useRenderToTexture=!1))},this.setRenderTargetFramebuffer=function(S,R){const D=De.get(S);D.__webglFramebuffer=R,D.__useDefaultFramebuffer=R===void 0},this.setRenderTarget=function(S,R=0,D=0){C=S,T=R,_=D;let N=!0,w=null,Ae=!1,de=!1;if(S){const Ie=De.get(S);Ie.__useDefaultFramebuffer!==void 0?(Se.bindFramebuffer(L.FRAMEBUFFER,null),N=!1):Ie.__webglFramebuffer===void 0?Re.setupRenderTarget(S):Ie.__hasExternalTextures&&Re.rebindTextures(S,De.get(S.texture).__webglTexture,De.get(S.depthTexture).__webglTexture);const ke=S.texture;(ke.isData3DTexture||ke.isDataArrayTexture||ke.isCompressedArrayTexture)&&(de=!0);const Ce=De.get(S).__webglFramebuffer;S.isWebGLCubeRenderTarget?(Array.isArray(Ce[R])?w=Ce[R][D]:w=Ce[R],Ae=!0):he.isWebGL2&&S.samples>0&&Re.useMultisampledRTT(S)===!1?w=De.get(S).__webglMultisampledFramebuffer:Array.isArray(Ce)?w=Ce[D]:w=Ce,M.copy(S.viewport),I.copy(S.scissor),H=S.scissorTest}else M.copy(j).multiplyScalar(G).floor(),I.copy(X).multiplyScalar(G).floor(),H=ee;if(Se.bindFramebuffer(L.FRAMEBUFFER,w)&&he.drawBuffers&&N&&Se.drawBuffers(S,w),Se.viewport(M),Se.scissor(I),Se.setScissorTest(H),Ae){const Ie=De.get(S.texture);L.framebufferTexture2D(L.FRAMEBUFFER,L.COLOR_ATTACHMENT0,L.TEXTURE_CUBE_MAP_POSITIVE_X+R,Ie.__webglTexture,D)}else if(de){const Ie=De.get(S.texture),ke=R||0;L.framebufferTextureLayer(L.FRAMEBUFFER,L.COLOR_ATTACHMENT0,Ie.__webglTexture,D||0,ke)}B=-1},this.readRenderTargetPixels=function(S,R,D,N,w,Ae,de){if(!(S&&S.isWebGLRenderTarget)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Ee=De.get(S).__webglFramebuffer;if(S.isWebGLCubeRenderTarget&&de!==void 0&&(Ee=Ee[de]),Ee){Se.bindFramebuffer(L.FRAMEBUFFER,Ee);try{const Ie=S.texture,ke=Ie.format,Ce=Ie.type;if(ke!==kt&&ce.convert(ke)!==L.getParameter(L.IMPLEMENTATION_COLOR_READ_FORMAT)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}const Te=Ce===ri&&(He.has("EXT_color_buffer_half_float")||he.isWebGL2&&He.has("EXT_color_buffer_float"));if(Ce!==nn&&ce.convert(Ce)!==L.getParameter(L.IMPLEMENTATION_COLOR_READ_TYPE)&&!(Ce===sn&&(he.isWebGL2||He.has("OES_texture_float")||He.has("WEBGL_color_buffer_float")))&&!Te){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}R>=0&&R<=S.width-N&&D>=0&&D<=S.height-w&&L.readPixels(R,D,N,w,ce.convert(ke),ce.convert(Ce),Ae)}finally{const Ie=C!==null?De.get(C).__webglFramebuffer:null;Se.bindFramebuffer(L.FRAMEBUFFER,Ie)}}},this.copyFramebufferToTexture=function(S,R,D=0){const N=Math.pow(2,-D),w=Math.floor(R.image.width*N),Ae=Math.floor(R.image.height*N);Re.setTexture2D(R,0),L.copyTexSubImage2D(L.TEXTURE_2D,D,0,0,S.x,S.y,w,Ae),Se.unbindTexture()},this.copyTextureToTexture=function(S,R,D,N=0){const w=R.image.width,Ae=R.image.height,de=ce.convert(D.format),Ee=ce.convert(D.type);Re.setTexture2D(D,0),L.pixelStorei(L.UNPACK_FLIP_Y_WEBGL,D.flipY),L.pixelStorei(L.UNPACK_PREMULTIPLY_ALPHA_WEBGL,D.premultiplyAlpha),L.pixelStorei(L.UNPACK_ALIGNMENT,D.unpackAlignment),R.isDataTexture?L.texSubImage2D(L.TEXTURE_2D,N,S.x,S.y,w,Ae,de,Ee,R.image.data):R.isCompressedTexture?L.compressedTexSubImage2D(L.TEXTURE_2D,N,S.x,S.y,R.mipmaps[0].width,R.mipmaps[0].height,de,R.mipmaps[0].data):L.texSubImage2D(L.TEXTURE_2D,N,S.x,S.y,de,Ee,R.image),N===0&&D.generateMipmaps&&L.generateMipmap(L.TEXTURE_2D),Se.unbindTexture()},this.copyTextureToTexture3D=function(S,R,D,N,w=0){if(v.isWebGL1Renderer){console.warn("THREE.WebGLRenderer.copyTextureToTexture3D: can only be used with WebGL2.");return}const Ae=S.max.x-S.min.x+1,de=S.max.y-S.min.y+1,Ee=S.max.z-S.min.z+1,Ie=ce.convert(N.format),ke=ce.convert(N.type);let Ce;if(N.isData3DTexture)Re.setTexture3D(N,0),Ce=L.TEXTURE_3D;else if(N.isDataArrayTexture||N.isCompressedArrayTexture)Re.setTexture2DArray(N,0),Ce=L.TEXTURE_2D_ARRAY;else{console.warn("THREE.WebGLRenderer.copyTextureToTexture3D: only supports THREE.DataTexture3D and THREE.DataTexture2DArray.");return}L.pixelStorei(L.UNPACK_FLIP_Y_WEBGL,N.flipY),L.pixelStorei(L.UNPACK_PREMULTIPLY_ALPHA_WEBGL,N.premultiplyAlpha),L.pixelStorei(L.UNPACK_ALIGNMENT,N.unpackAlignment);const Te=L.getParameter(L.UNPACK_ROW_LENGTH),$e=L.getParameter(L.UNPACK_IMAGE_HEIGHT),It=L.getParameter(L.UNPACK_SKIP_PIXELS),at=L.getParameter(L.UNPACK_SKIP_ROWS),Kt=L.getParameter(L.UNPACK_SKIP_IMAGES),Ve=D.isCompressedTexture?D.mipmaps[w]:D.image;L.pixelStorei(L.UNPACK_ROW_LENGTH,Ve.width),L.pixelStorei(L.UNPACK_IMAGE_HEIGHT,Ve.height),L.pixelStorei(L.UNPACK_SKIP_PIXELS,S.min.x),L.pixelStorei(L.UNPACK_SKIP_ROWS,S.min.y),L.pixelStorei(L.UNPACK_SKIP_IMAGES,S.min.z),D.isDataTexture||D.isData3DTexture?L.texSubImage3D(Ce,w,R.x,R.y,R.z,Ae,de,Ee,Ie,ke,Ve.data):D.isCompressedArrayTexture?(console.warn("THREE.WebGLRenderer.copyTextureToTexture3D: untested support for compressed srcTexture."),L.compressedTexSubImage3D(Ce,w,R.x,R.y,R.z,Ae,de,Ee,Ie,Ve.data)):L.texSubImage3D(Ce,w,R.x,R.y,R.z,Ae,de,Ee,Ie,ke,Ve),L.pixelStorei(L.UNPACK_ROW_LENGTH,Te),L.pixelStorei(L.UNPACK_IMAGE_HEIGHT,$e),L.pixelStorei(L.UNPACK_SKIP_PIXELS,It),L.pixelStorei(L.UNPACK_SKIP_ROWS,at),L.pixelStorei(L.UNPACK_SKIP_IMAGES,Kt),w===0&&N.generateMipmaps&&L.generateMipmap(Ce),Se.unbindTexture()},this.initTexture=function(S){S.isCubeTexture?Re.setTextureCube(S,0):S.isData3DTexture?Re.setTexture3D(S,0):S.isDataArrayTexture||S.isCompressedArrayTexture?Re.setTexture2DArray(S,0):Re.setTexture2D(S,0),Se.unbindTexture()},this.resetState=function(){T=0,_=0,C=null,Se.reset(),ue.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return jt}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(t){this._outputColorSpace=t;const e=this.getContext();e.drawingBufferColorSpace=t===Tr?"display-p3":"srgb",e.unpackColorSpace=je.workingColorSpace===Ii?"display-p3":"srgb"}get outputEncoding(){return console.warn("THREE.WebGLRenderer: Property .outputEncoding has been removed. Use .outputColorSpace instead."),this.outputColorSpace===ot?gn:ra}set outputEncoding(t){console.warn("THREE.WebGLRenderer: Property .outputEncoding has been removed. Use .outputColorSpace instead."),this.outputColorSpace=t===gn?ot:zt}get useLegacyLights(){return console.warn("THREE.WebGLRenderer: The property .useLegacyLights has been deprecated. Migrate your lighting according to the following guide: https://discourse.threejs.org/t/updates-to-lighting-in-three-js-r155/53733."),this._useLegacyLights}set useLegacyLights(t){console.warn("THREE.WebGLRenderer: The property .useLegacyLights has been deprecated. Migrate your lighting according to the following guide: https://discourse.threejs.org/t/updates-to-lighting-in-three-js-r155/53733."),this._useLegacyLights=t}},ch=class extends mo{};ch.prototype.isWebGL1Renderer=!0;var uh=class extends Jt{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(t,e){return super.copy(t,e),t.background!==null&&(this.background=t.background.clone()),t.environment!==null&&(this.environment=t.environment.clone()),t.fog!==null&&(this.fog=t.fog.clone()),this.backgroundBlurriness=t.backgroundBlurriness,this.backgroundIntensity=t.backgroundIntensity,t.overrideMaterial!==null&&(this.overrideMaterial=t.overrideMaterial.clone()),this.matrixAutoUpdate=t.matrixAutoUpdate,this}toJSON(t){const e=super.toJSON(t);return this.fog!==null&&(e.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(e.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(e.object.backgroundIntensity=this.backgroundIntensity),e}},dh=class extends fi{constructor(t){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new Ne(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.alphaMap=t.alphaMap,this.size=t.size,this.sizeAttenuation=t.sizeAttenuation,this.fog=t.fog,this}},Eo=new gt,is=new Ea,nr=new wi,ir=new J,fh=class extends Jt{constructor(t=new un,e=new dh){super(),this.isPoints=!0,this.type="Points",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}raycast(t,e){const n=this.geometry,i=this.matrixWorld,r=t.params.Points.threshold,s=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),nr.copy(n.boundingSphere),nr.applyMatrix4(i),nr.radius+=r,t.ray.intersectsSphere(nr)===!1)return;Eo.copy(i).invert(),is.copy(t.ray).applyMatrix4(Eo);const o=r/((this.scale.x+this.scale.y+this.scale.z)/3),a=o*o,A=n.index,c=n.attributes.position;if(A!==null){const d=Math.max(0,s.start),f=Math.min(A.count,s.start+s.count);for(let p=d,E=f;p<E;p++){const g=A.getX(p);ir.fromBufferAttribute(c,g),So(ir,g,a,i,t,e,this)}}else{const d=Math.max(0,s.start),f=Math.min(c.count,s.start+s.count);for(let p=d,E=f;p<E;p++)ir.fromBufferAttribute(c,p),So(ir,p,a,i,t,e,this)}}updateMorphTargets(){const e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){const i=e[n[0]];if(i!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,s=i.length;r<s;r++){const o=i[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}};function So(t,e,n,i,r,s,o){const a=is.distanceSqToPoint(t);if(a<n){const A=new J;is.closestPointToPoint(t,A),A.applyMatrix4(i);const l=r.ray.origin.distanceTo(A);if(l<r.near||l>r.far)return;s.push({distance:l,distanceToRay:Math.sqrt(a),point:A,index:e,face:null,object:o})}}var hh=class extends wt{constructor(t,e,n,i,r,s,o,a,A){super(t,e,n,i,r,s,o,a,A),this.isCanvasTexture=!0,this.needsUpdate=!0}},ph=class Jo extends un{constructor(e=1,n=32,i=16,r=0,s=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:n,heightSegments:i,phiStart:r,phiLength:s,thetaStart:o,thetaLength:a},n=Math.max(3,Math.floor(n)),i=Math.max(2,Math.floor(i));const A=Math.min(o+a,Math.PI);let l=0;const c=[],d=new J,f=new J,p=[],E=[],g=[],h=[];for(let u=0;u<=i;u++){const v=[],y=u/i;let T=0;u===0&&o===0?T=.5/n:u===i&&A===Math.PI&&(T=-.5/n);for(let _=0;_<=n;_++){const C=_/n;d.x=-e*Math.cos(r+C*s)*Math.sin(o+y*a),d.y=e*Math.cos(o+y*a),d.z=e*Math.sin(r+C*s)*Math.sin(o+y*a),E.push(d.x,d.y,d.z),f.copy(d).normalize(),g.push(f.x,f.y,f.z),h.push(C+T,1-y),v.push(l++)}c.push(v)}for(let u=0;u<i;u++)for(let v=0;v<n;v++){const y=c[u][v+1],T=c[u][v],_=c[u+1][v],C=c[u+1][v+1];(u!==0||o>0)&&p.push(y,T,C),(u!==i-1||A<Math.PI)&&p.push(T,_,C)}this.setIndex(p),this.setAttribute("position",new mt(E,3)),this.setAttribute("normal",new mt(g,3)),this.setAttribute("uv",new mt(h,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Jo(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}};function rr(t,e,n){return!t||!n&&t.constructor===e?t:typeof e.BYTES_PER_ELEMENT=="number"?new e(t):Array.prototype.slice.call(t)}function gh(t){return ArrayBuffer.isView(t)&&!(t instanceof DataView)}var sr=class{constructor(t,e,n,i){this.parameterPositions=t,this._cachedIndex=0,this.resultBuffer=i!==void 0?i:new e.constructor(n),this.sampleValues=e,this.valueSize=n,this.settings=null,this.DefaultSettings_={}}evaluate(t){const e=this.parameterPositions;let n=this._cachedIndex,i=e[n],r=e[n-1];n:{e:{let s;t:{i:if(!(t<i)){for(let o=n+2;;){if(i===void 0){if(t<r)break i;return n=e.length,this._cachedIndex=n,this.copySampleValue_(n-1)}if(n===o)break;if(r=i,i=e[++n],t<i)break e}s=e.length;break t}if(!(t>=r)){const o=e[1];t<o&&(n=2,r=o);for(let a=n-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(n===a)break;if(i=r,r=e[--n-1],t>=r)break e}s=n,n=0;break t}break n}for(;n<s;){const o=n+s>>>1;t<e[o]?s=o:n=o+1}if(i=e[n],r=e[n-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(i===void 0)return n=e.length,this._cachedIndex=n,this.copySampleValue_(n-1)}this._cachedIndex=n,this.intervalChanged_(n,r,i)}return this.interpolate_(n,r,t,i)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(t){const e=this.resultBuffer,n=this.sampleValues,i=this.valueSize,r=t*i;for(let s=0;s!==i;++s)e[s]=n[r+s];return e}interpolate_(){throw new Error("call to abstract method")}intervalChanged_(){}},mh=class extends sr{constructor(t,e,n,i){super(t,e,n,i),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:ta,endingEnd:ta}}intervalChanged_(t,e,n){const i=this.parameterPositions;let r=t-2,s=t+1,o=i[r],a=i[s];if(o===void 0)switch(this.getSettings_().endingStart){case na:r=t,o=2*e-n;break;case ia:r=i.length-2,o=e+i[r]-i[r+1];break;default:r=t,o=n}if(a===void 0)switch(this.getSettings_().endingEnd){case na:s=t,a=2*n-e;break;case ia:s=1,a=n+i[1]-i[0];break;default:s=t-1,a=e}const A=(n-e)*.5,l=this.valueSize;this._weightPrev=A/(e-o),this._weightNext=A/(a-n),this._offsetPrev=r*l,this._offsetNext=s*l}interpolate_(t,e,n,i){const r=this.resultBuffer,s=this.sampleValues,o=this.valueSize,a=t*o,A=a-o,l=this._offsetPrev,c=this._offsetNext,d=this._weightPrev,f=this._weightNext,p=(n-e)/(i-e),E=p*p,g=E*p,h=-d*g+2*d*E-d*p,u=(1+d)*g+(-1.5-2*d)*E+(-.5+d)*p+1,v=(-1-f)*g+(1.5+f)*E+.5*p,y=f*g-f*E;for(let T=0;T!==o;++T)r[T]=h*s[l+T]+u*s[A+T]+v*s[a+T]+y*s[c+T];return r}},Eh=class extends sr{constructor(t,e,n,i){super(t,e,n,i)}interpolate_(t,e,n,i){const r=this.resultBuffer,s=this.sampleValues,o=this.valueSize,a=t*o,A=a-o,l=(n-e)/(i-e),c=1-l;for(let d=0;d!==o;++d)r[d]=s[A+d]*c+s[a+d]*l;return r}},Sh=class extends sr{constructor(t,e,n,i){super(t,e,n,i)}interpolate_(t){return this.copySampleValue_(t-1)}},qt=class{constructor(t,e,n,i){if(t===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(e===void 0||e.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+t);this.name=t,this.times=rr(e,this.TimeBufferType),this.values=rr(n,this.ValueBufferType),this.setInterpolation(i||this.DefaultInterpolation)}static toJSON(t){const e=t.constructor;let n;if(e.toJSON!==this.toJSON)n=e.toJSON(t);else{n={name:t.name,times:rr(t.times,Array),values:rr(t.values,Array)};const i=t.getInterpolation();i!==t.DefaultInterpolation&&(n.interpolation=i)}return n.type=t.ValueTypeName,n}InterpolantFactoryMethodDiscrete(t){return new Sh(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodLinear(t){return new Eh(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodSmooth(t){return new mh(this.times,this.values,this.getValueSize(),t)}setInterpolation(t){let e;switch(t){case xi:e=this.InterpolantFactoryMethodDiscrete;break;case Mi:e=this.InterpolantFactoryMethodLinear;break;case Cr:e=this.InterpolantFactoryMethodSmooth;break}if(e===void 0){const n="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(t!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(n);return console.warn("THREE.KeyframeTrack:",n),this}return this.createInterpolant=e,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return xi;case this.InterpolantFactoryMethodLinear:return Mi;case this.InterpolantFactoryMethodSmooth:return Cr}}getValueSize(){return this.values.length/this.times.length}shift(t){if(t!==0){const e=this.times;for(let n=0,i=e.length;n!==i;++n)e[n]+=t}return this}scale(t){if(t!==1){const e=this.times;for(let n=0,i=e.length;n!==i;++n)e[n]*=t}return this}trim(t,e){const n=this.times,i=n.length;let r=0,s=i-1;for(;r!==i&&n[r]<t;)++r;for(;s!==-1&&n[s]>e;)--s;if(++s,r!==0||s!==i){r>=s&&(s=Math.max(s,1),r=s-1);const o=this.getValueSize();this.times=n.slice(r,s),this.values=this.values.slice(r*o,s*o)}return this}validate(){let t=!0;const e=this.getValueSize();e-Math.floor(e)!==0&&(console.error("THREE.KeyframeTrack: Invalid value size in track.",this),t=!1);const n=this.times,i=this.values,r=n.length;r===0&&(console.error("THREE.KeyframeTrack: Track is empty.",this),t=!1);let s=null;for(let o=0;o!==r;o++){const a=n[o];if(typeof a=="number"&&isNaN(a)){console.error("THREE.KeyframeTrack: Time is not a valid number.",this,o,a),t=!1;break}if(s!==null&&s>a){console.error("THREE.KeyframeTrack: Out of order keys.",this,o,a,s),t=!1;break}s=a}if(i!==void 0&&gh(i))for(let o=0,a=i.length;o!==a;++o){const A=i[o];if(isNaN(A)){console.error("THREE.KeyframeTrack: Value is not a valid number.",this,o,A),t=!1;break}}return t}optimize(){const t=this.times.slice(),e=this.values.slice(),n=this.getValueSize(),i=this.getInterpolation()===Cr,r=t.length-1;let s=1;for(let o=1;o<r;++o){let a=!1;const A=t[o],l=t[o+1];if(A!==l&&(o!==1||A!==t[0]))if(i)a=!0;else{const c=o*n,d=c-n,f=c+n;for(let p=0;p!==n;++p){const E=e[c+p];if(E!==e[d+p]||E!==e[f+p]){a=!0;break}}}if(a){if(o!==s){t[s]=t[o];const c=o*n,d=s*n;for(let f=0;f!==n;++f)e[d+f]=e[c+f]}++s}}if(r>0){t[s]=t[r];for(let o=r*n,a=s*n,A=0;A!==n;++A)e[a+A]=e[o+A];++s}return s!==t.length?(this.times=t.slice(0,s),this.values=e.slice(0,s*n)):(this.times=t,this.values=e),this}clone(){const t=this.times.slice(),e=this.values.slice(),n=this.constructor,i=new n(this.name,t,e);return i.createInterpolant=this.createInterpolant,i}};qt.prototype.TimeBufferType=Float32Array,qt.prototype.ValueBufferType=Float32Array,qt.prototype.DefaultInterpolation=Mi;var gi=class extends qt{};gi.prototype.ValueTypeName="bool",gi.prototype.ValueBufferType=Array,gi.prototype.DefaultInterpolation=xi,gi.prototype.InterpolantFactoryMethodLinear=void 0,gi.prototype.InterpolantFactoryMethodSmooth=void 0;var vh=class extends qt{};vh.prototype.ValueTypeName="color";var xh=class extends qt{};xh.prototype.ValueTypeName="number";var Mh=class extends sr{constructor(t,e,n,i){super(t,e,n,i)}interpolate_(t,e,n,i){const r=this.resultBuffer,s=this.sampleValues,o=this.valueSize,a=(n-e)/(i-e);let A=t*o;for(let l=A+o;A!==l;A+=4)Dn.slerpFlat(r,0,s,A-o,s,A,a);return r}},rs=class extends qt{InterpolantFactoryMethodLinear(t){return new Mh(this.times,this.values,this.getValueSize(),t)}};rs.prototype.ValueTypeName="quaternion",rs.prototype.DefaultInterpolation=Mi,rs.prototype.InterpolantFactoryMethodSmooth=void 0;var mi=class extends qt{};mi.prototype.ValueTypeName="string",mi.prototype.ValueBufferType=Array,mi.prototype.DefaultInterpolation=xi,mi.prototype.InterpolantFactoryMethodLinear=void 0,mi.prototype.InterpolantFactoryMethodSmooth=void 0;var Ih=class extends qt{};Ih.prototype.ValueTypeName="vector";var yh=class{constructor(t,e,n){const i=this;let r=!1,s=0,o=0,a;const A=[];this.onStart=void 0,this.onLoad=t,this.onProgress=e,this.onError=n,this.itemStart=function(l){o++,r===!1&&i.onStart!==void 0&&i.onStart(l,s,o),r=!0},this.itemEnd=function(l){s++,i.onProgress!==void 0&&i.onProgress(l,s,o),s===o&&(r=!1,i.onLoad!==void 0&&i.onLoad())},this.itemError=function(l){i.onError!==void 0&&i.onError(l)},this.resolveURL=function(l){return a?a(l):l},this.setURLModifier=function(l){return a=l,this},this.addHandler=function(l,c){return A.push(l,c),this},this.removeHandler=function(l){const c=A.indexOf(l);return c!==-1&&A.splice(c,2),this},this.getHandler=function(l){for(let c=0,d=A.length;c<d;c+=2){const f=A[c],p=A[c+1];if(f.global&&(f.lastIndex=0),f.test(l))return p}return null}}},Ch=new yh,Th=class{constructor(t){this.manager=t!==void 0?t:Ch,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={}}load(){}loadAsync(t,e){const n=this;return new Promise(function(i,r){n.load(t,i,e,r)})}parse(){}setCrossOrigin(t){return this.crossOrigin=t,this}setWithCredentials(t){return this.withCredentials=t,this}setPath(t){return this.path=t,this}setResourcePath(t){return this.resourcePath=t,this}setRequestHeader(t){return this.requestHeader=t,this}};Th.DEFAULT_MATERIAL_NAME="__DEFAULT";var Ph=class{constructor(t=!0){this.autoStart=t,this.startTime=0,this.oldTime=0,this.elapsedTime=0,this.running=!1}start(){this.startTime=vo(),this.oldTime=this.startTime,this.elapsedTime=0,this.running=!0}stop(){this.getElapsedTime(),this.running=!1,this.autoStart=!1}getElapsedTime(){return this.getDelta(),this.elapsedTime}getDelta(){let t=0;if(this.autoStart&&!this.running)return this.start(),0;if(this.running){const e=vo();t=(e-this.oldTime)/1e3,this.oldTime=e,this.elapsedTime+=t}return t}};function vo(){return(typeof performance>"u"?Date:performance).now()}var ss="\\[\\]\\.:\\/",bh=new RegExp("["+ss+"]","g"),as="[^"+ss+"]",Bh="[^"+ss.replace("\\.","")+"]",kh=/((?:WC+[\/:])*)/.source.replace("WC",as),Rh=/(WCOD+)?/.source.replace("WCOD",Bh),_h=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",as),Lh=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",as),wh=new RegExp("^"+kh+Rh+_h+Lh+"$"),Jh=["material","materials","bones","map"],Oh=class{constructor(t,e,n){const i=n||qe.parseTrackName(e);this._targetGroup=t,this._bindings=t.subscribe_(e,i)}getValue(t,e){this.bind();const n=this._targetGroup.nCachedObjects_,i=this._bindings[n];i!==void 0&&i.getValue(t,e)}setValue(t,e){const n=this._bindings;for(let i=this._targetGroup.nCachedObjects_,r=n.length;i!==r;++i)n[i].setValue(t,e)}bind(){const t=this._bindings;for(let e=this._targetGroup.nCachedObjects_,n=t.length;e!==n;++e)t[e].bind()}unbind(){const t=this._bindings;for(let e=this._targetGroup.nCachedObjects_,n=t.length;e!==n;++e)t[e].unbind()}},qe=class ni{constructor(e,n,i){this.path=n,this.parsedPath=i||ni.parseTrackName(n),this.node=ni.findNode(e,this.parsedPath.nodeName),this.rootNode=e,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(e,n,i){return e&&e.isAnimationObjectGroup?new ni.Composite(e,n,i):new ni(e,n,i)}static sanitizeNodeName(e){return e.replace(/\s/g,"_").replace(bh,"")}static parseTrackName(e){const n=wh.exec(e);if(n===null)throw new Error("PropertyBinding: Cannot parse trackName: "+e);const i={nodeName:n[2],objectName:n[3],objectIndex:n[4],propertyName:n[5],propertyIndex:n[6]},r=i.nodeName&&i.nodeName.lastIndexOf(".");if(r!==void 0&&r!==-1){const s=i.nodeName.substring(r+1);Jh.indexOf(s)!==-1&&(i.nodeName=i.nodeName.substring(0,r),i.objectName=s)}if(i.propertyName===null||i.propertyName.length===0)throw new Error("PropertyBinding: can not parse propertyName from trackName: "+e);return i}static findNode(e,n){if(n===void 0||n===""||n==="."||n===-1||n===e.name||n===e.uuid)return e;if(e.skeleton){const i=e.skeleton.getBoneByName(n);if(i!==void 0)return i}if(e.children){const i=function(s){for(let o=0;o<s.length;o++){const a=s[o];if(a.name===n||a.uuid===n)return a;const A=i(a.children);if(A)return A}return null},r=i(e.children);if(r)return r}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(e,n){e[n]=this.targetObject[this.propertyName]}_getValue_array(e,n){const i=this.resolvedProperty;for(let r=0,s=i.length;r!==s;++r)e[n++]=i[r]}_getValue_arrayElement(e,n){e[n]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(e,n){this.resolvedProperty.toArray(e,n)}_setValue_direct(e,n){this.targetObject[this.propertyName]=e[n]}_setValue_direct_setNeedsUpdate(e,n){this.targetObject[this.propertyName]=e[n],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(e,n){this.targetObject[this.propertyName]=e[n],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(e,n){const i=this.resolvedProperty;for(let r=0,s=i.length;r!==s;++r)i[r]=e[n++]}_setValue_array_setNeedsUpdate(e,n){const i=this.resolvedProperty;for(let r=0,s=i.length;r!==s;++r)i[r]=e[n++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(e,n){const i=this.resolvedProperty;for(let r=0,s=i.length;r!==s;++r)i[r]=e[n++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(e,n){this.resolvedProperty[this.propertyIndex]=e[n]}_setValue_arrayElement_setNeedsUpdate(e,n){this.resolvedProperty[this.propertyIndex]=e[n],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(e,n){this.resolvedProperty[this.propertyIndex]=e[n],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(e,n){this.resolvedProperty.fromArray(e,n)}_setValue_fromArray_setNeedsUpdate(e,n){this.resolvedProperty.fromArray(e,n),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(e,n){this.resolvedProperty.fromArray(e,n),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(e,n){this.bind(),this.getValue(e,n)}_setValue_unbound(e,n){this.bind(),this.setValue(e,n)}bind(){let e=this.node;const n=this.parsedPath,i=n.objectName,r=n.propertyName;let s=n.propertyIndex;if(e||(e=ni.findNode(this.rootNode,n.nodeName),this.node=e),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!e){console.warn("THREE.PropertyBinding: No target node found for track: "+this.path+".");return}if(i){let l=n.objectIndex;switch(i){case"materials":if(!e.material){console.error("THREE.PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.materials){console.error("THREE.PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);return}e=e.material.materials;break;case"bones":if(!e.skeleton){console.error("THREE.PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);return}e=e.skeleton.bones;for(let c=0;c<e.length;c++)if(e[c].name===l){l=c;break}break;case"map":if("map"in e){e=e.map;break}if(!e.material){console.error("THREE.PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!e.material.map){console.error("THREE.PropertyBinding: Can not bind to material.map as node.material does not have a map.",this);return}e=e.material.map;break;default:if(e[i]===void 0){console.error("THREE.PropertyBinding: Can not bind to objectName of node undefined.",this);return}e=e[i]}if(l!==void 0){if(e[l]===void 0){console.error("THREE.PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,e);return}e=e[l]}}const o=e[r];if(o===void 0){const l=n.nodeName;console.error("THREE.PropertyBinding: Trying to update property for track: "+l+"."+r+" but it wasn't found.",e);return}let a=this.Versioning.None;this.targetObject=e,e.needsUpdate!==void 0?a=this.Versioning.NeedsUpdate:e.matrixWorldNeedsUpdate!==void 0&&(a=this.Versioning.MatrixWorldNeedsUpdate);let A=this.BindingType.Direct;if(s!==void 0){if(r==="morphTargetInfluences"){if(!e.geometry){console.error("THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);return}if(!e.geometry.morphAttributes){console.error("THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);return}e.morphTargetDictionary[s]!==void 0&&(s=e.morphTargetDictionary[s])}A=this.BindingType.ArrayElement,this.resolvedProperty=o,this.propertyIndex=s}else o.fromArray!==void 0&&o.toArray!==void 0?(A=this.BindingType.HasFromToArray,this.resolvedProperty=o):Array.isArray(o)?(A=this.BindingType.EntireArray,this.resolvedProperty=o):this.propertyName=r;this.getValue=this.GetterByBindingType[A],this.setValue=this.SetterByBindingTypeAndVersioning[A][a]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};qe.Composite=Oh,qe.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3},qe.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2},qe.prototype.GetterByBindingType=[qe.prototype._getValue_direct,qe.prototype._getValue_array,qe.prototype._getValue_arrayElement,qe.prototype._getValue_toArray],qe.prototype.SetterByBindingTypeAndVersioning=[[qe.prototype._setValue_direct,qe.prototype._setValue_direct_setNeedsUpdate,qe.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[qe.prototype._setValue_array,qe.prototype._setValue_array_setNeedsUpdate,qe.prototype._setValue_array_setMatrixWorldNeedsUpdate],[qe.prototype._setValue_arrayElement,qe.prototype._setValue_arrayElement_setNeedsUpdate,qe.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[qe.prototype._setValue_fromArray,qe.prototype._setValue_fromArray_setNeedsUpdate,qe.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];var Vh=new Float32Array(1);typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:ur}})),typeof window<"u"&&(window.__THREE__?console.warn("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=ur);var Dh="data:image/jpeg;base64,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SAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIAdjEyNRBJ7gLQDG39W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Z/T7vTVud8tg88knrq9U8EZdPKXi6XWA5GUoSj1BDzXFgIukV/gwBZx9Z/ItnFj9x+TrXquniC/ayIdYUFEsUTenVtMe5xdY0DglHaabM0ZWTX06DueS3qCpMAEgBIASAEgBIASAEgBIASAEgBIASAEgGgXfsgaQDHdNdfwQDEgBIASAEgBIASAEgBlt01+iARSAEgBIASAEgBIASAHTXZfQde/tQDEgBIASAEgBIASAEgBIASAEgBIASAEgGjbRu77K6fNxACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJAJQnLGd0SYnvDFS46APU4/KjlgMZPnI13ACzZ+0j5d2r5YNEEdhv6PbXfvmL1cToPWyR2SINWCQaII69hGhHuNGUpGfnP7WvxvV7rfAERR66htABjeg7NGN9Z4CMxRw4M+OefEORi+6eEZTDeNRslONyge3vqmEnltLc4uL54/h4tWcr8eRTnx4jORwCcIE6QmdxHsZUL9jTZLGBHzboyO0x00MTepN33Vob1efZ78tF9hzRsEAxGpGt0B8a7HZ/dTjmNXqg6/UcfG4/KzY+PnjysImfDzRhOAnE6giOQRmOteYA6PHLvc629v5pi+S1bjPFyiPamAGodXQAsqIFi7q9a617LNAIgdpZARBPdZqxrXupFmPEEgO50UXQCR9nTd/u9/ghRlxP4dOn4q/wCPggEqj0Mh9t35jrX29OvZ3BiEAkCew/O/0ubqlfm9iKux0PyQolt8QgRucpHaIgG9K21r293Y7A49uXfCUpyj5DEgATMgTKY2ky8tgAEUTdsEVX39Pn2a9jKN/dtvaY+8f/IH+4+9NRBDqdNfkiL7WoAZWbAodg69ewXfy7W6HHy5cc544bo4jc9upiOokY9RHT7qodCejWbtJcW9eguLXL2nvbIYpz3VGtsPEJkRHyWBetXZOlalJnH1wght7bB16dvx+DaTHd5YnQDQntA1N1Hts11bgBCQEY321pqBqTpdg3707yBswzEoxvSOpO4HQ3QNHuvUa97NmbyDgmQZEjpbFACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQ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T4zn8FLri4zL8BfkyivKdfw+TQ3KAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkA9bn/ANQc31XwBzJQmOPijhw+HixYRDFACMYmOKMRLaBpKVz7yXyXOmk9POPHry0u2126+COmfJGlebTqfy73mddzILv1mXURjfYTrVG7AOn1BDS23KAmckj2/gAfwYIBPxZ0RfX21+rBZAXcrl8jm5fG5GWWbJthDfP7jHHAQgCfaIA7+9pZJNeJwpbb1BIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAaDRtxALBNrQDpjO3mblEVfLP2DVoblEVu6X9x+riAaJSHafq4gG7pd5+riAEgBIASAEgBIASAEgBIBfHNth1shoblAWY7lMSOuvUsRtrW2oD0ePxcmfhZ+buxjHgz4sGzePFnPNHJKOzH1MYxgTOXQWO98+OSURQ6XfzXf+aa+NzfoGOM+Q7vi1Qybo/m9WZRFu7WqYmQqnWU7oqJS6k1Q7tf06sd2jUyqJ7vcNd2fi6yxkEpSB7Gs3Vntuq9m7VjKo3r0s/uHU/LtYGoxvdrZG3W6rrfSj0q22sg0LHuuOSJoxkDE90hrfyLVmuf0RonCIFkka9B1+i5nIy8jNly5pb8mScsk56eacySToK1JJ0VnCWY/LOkIdXZ6p696jzuFwvTMvLzZeHwheLjS2eFiymxI49upG2tZE2TIvkPPXTWbbbYmb4+zbV2tkmeIyJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAEEdQ2Y5C6lqPzHdaBFbshtkR9PgkUYmgCQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAnhoTjI0dpEqPbtN18+jGPVlUHbPNLJ4mWREZZZGUowGwHeSTQjUREGqiPaujWRDw41ukesvLoKOlUTempumSYxPJS0dWM+Hhxkd/uOz8SxjWwixX3Cunw+L31401+KTp+MZVI7Sdb77/AMHaB/aqh793Tp8u5CiFX01+XZ3vRHFHLjlM8jDCe6MRikMgM4m7mJCMoARNXukOunRJbZji2efH/YfOOUqUa7te4g/8KiCDpFOVBu4mtSaFCz0HcO4ezDUMQE+vU1/Htq7E4wJHJKcfKduyIkN+lCdyjUSLsiyDWjUuYCJ/j3azmh3n6IyC8zGSQ3AQjYvZGOnQEgEizQuiat55ciPYP4+ZK/fKZFW0T0BOtdO/pp3tB5MriYDYQBr1NjtGgr+NWplFSzTOMTxVKM72zuxoDewx79wHwponOeWcpzlKc5yMpSkSZSlI2ZSJ1JJ1JLLyAikAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAsxSrRrQlV067qOhecSIblEV06m7afGPaA1EMO7jcfNy82LjYBGeTPKMYxsR89kCJM6APwNavF+sZBGcAQI5ABIUNQDYGuo1HYrcc3wOqznieJ0ej636Rz/AEDmS4fPweDmEYzAE4TjPHMXHJDJAyjOEh0lEkHseTL6hn5Axxz5J5I4cccWKJJIx4o9IQB6RHcHOvqa+pM6884+Fngs1mucTrc33q7a3W8pbb49FMbPZduTzSlfQX3AAaewADQG7zEtSyAnKZl9bYLOQBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIALsYmZoV8zSAehmzn1DDxycOCB4uEYpZMUIwllhuO2WUXU5RvbvAsit3e8kLA2jq5msmeb+a5582i3PyRCeOWOroiQsURLS+0AkxPsaLZKRNVpQ+Fdp6dlsyqopAJZylGqGp70KitIUEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASASxi5R0EtRodAfY6j83YaaoBdtInV9OpBr2sf4OxIlOcsYltH9xBIB76rRusNM/RCrPh17f49+rI67dACBRrodfu+LsmPD9/AD8XDpbRRO2F17NQEjqx1aiKdWOcfyz17Em/RFaQ8kck4jQmvqsMZqK6MxIxkfxReaeSc/ukT/h0bejOUVFIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBmD8/ikBBvEYES0o/s109+tmu5pgFJrR2fYgEUgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBuxQkbAjZ+HSkslvSZEVAkag03TxCIkSNumnbr8ff8GL22KhhIkaNR/wB2tfMAE9e5zxwIY4jHGJhuuQ3bpykesrJGg0FAaBiY5yKmcnhiVVcomPYdJCj19u0POTZt1URWJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQCV+UhxgDp44MY3VbrF/T/AJbMAqAPSjoKsGxqXppxM+a6dP4eaIAjSu7U+7pBuzWvw/QVKvKjQb0cHXvQoltJZRJ+DcEBm0BSNIoMlUYSkSBQ08wBs9oj+1Xc83JkNB29WbXDO4Kzk8pjXXt6fg1uc8YQBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIBocQC7HkjEHcD0NbSB5uy7B072lZAaTZcQAkAJACQAkAJACQAkAJACQAkAJACQAkAPTg4niDdOWyJ6d5Tevp55txBLXOASaAsvpYxHCSMcKP7Mj1Lh3mNbcT51WernhwpxG7L5B3ftn4R/SWzPllXnJNXoe8vOeleu3E/Fd9r4rkbjx5JiYwwnIY8c8k9kTLbjhrKciOkR2noHjOfIbqW2+yOjLvNZjOPBjOTAjPIZntrsDFZyKFuUgGpACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAnAEyA9/47n0vTPDhCcpbd0hQsHs7+xTq7+hJJbxyM7InEQNZA0B1sAHt1qu3vDYIy3Tkce8SPQn6bR/BZdbjrPm1i5t7cy+f8IZHL0P8AH+LdPYTUYCAvUAk6+7zauPCYaSICj/HY7qNWKokDGxZIF6y23Q767a7nQfr1vuN9bZbiKDz8uUylKpExs1pVi9CRZ1PdZejJxoyNg7XldrW7pAcbLJHZIxu6ea2YoIpgAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAswbBMGZoBrda4zz0ZSq7IZT5jHXd1sX06Gux5ASOlh339fdhFd08wkTKUQNP2dBfeXhJJel3l5s+jmip5p7yKNivca93ya1bkASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgFkIGdACyXcOTwiJaHXp3tkzwuu3bciV6PB4kMsN0rEo9NuhH6P0tvBNQyTGszdQvrX4vX09JZm9WvT6W+PkztTZCWTdIiZlDTS43oDqQb+6/dqO8mRHi7bs3Va/joe7ortnrx5cfvli5z/tj8FkWI/tH86/5W6u5go0e7m4drQRImnmyciO0iN339OqYu0wqGTldkP+79weVXfyYUCbNlIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAaASQB1OiBog9xtdScAlkwzxS2yBv8/dsnzM2Q3Ig9eyuvZYINezdtbrcWNX1Nr1x+/ghjChOAUSAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAY6gBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASATxiz0JrsDEGjpo2IDv4OaUJS75RlXx66dnY8+CQEo3Uh+XvWlvb0trn49GNLzOlZ2i1f4pNgzJ9tfyZZo1coEGxoNhJ/cHXd71redbP0/eBIhbXLdAeaNGwNpIB17a6uEudesVUyN0ZCwPKevb7DTqWujLS6Pt+ltZ6ormbMsdp637/wCLlaCtMAEgBIASAF0QCfhn593axsjtbhASlj2/tBg3CAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAOkVWoNi9D0+PulwDEwASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgGgkdLcQDrjyzIRBAB7ZA0T/AKj0+ryPSer0/X+7mmFegcf6ybMok1/dC707qt4Y5JR6Eh7XX7njnjzn8HKWzoz0adw9PygdBfeCD17xofj8GjHzcmOGwCJ7jWv+L1+zWdfVusxwz3LhdPDOA1iJjXSifa/l7HRplyyTcYiB6dAQR8+h+DdvTvx+DN9TxkwZMKp4ZRF0a93JTkTd/TQfRzdbC1RCkwASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAXYxglpOWTGe+hKN+9UR+LS6nbets/FlOVWZsJwkAmMhKO6MomwQ1t217ffPkiS5USAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgAC9EgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIAdlp33WtiqS0GJgAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJADKOyju3e22vxv9CXgEXdK7bv8GAMSAEgBIASAEgB3caq9O5GQYkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJADfh4mfOCYQJAF7joNO4nqmtdNtukEtkU7TV6fXX6dab48cjHKZnsN7fDFmc+8UDf1ctTXjOceGPGqmVUYwPWRuuwD9JH4PZxfTcuezIeHAjyyPv3DqdO9nDenpXb2is3bDkkcIPkE59NZ6X3+WP/wCT6w9J42KjmzGtepjAHu1txe3wzfi7fZ1n+W38F5Z77ekeXLlZSNoMYDuhGMfyFvrmfDxSMcM+NinGhc4X9J3rp3PLvvw+DtnSXGt1lnnP4tYjHPjl4kMc8ktsYknu/wCXu5880MkJ+LjkaO2WMjcBfeADTwkt4jp6l2llzL5WdXRnXHk4pYskPuhOPxiQ7kz5cpJlOR3ddaB+QoPOyzwpdretaymIrTBQSAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBlLHOH3QlH/AFRI/NLZZ1lgIpgAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQA7GMpyEYgykegHUokz0Bj6g9JgBGOTNtyz6RABHv7n4p2+zOM7c3wGO/24eZGO+QiKFmtTQ+ZL62L0iciRnlUQPL4ZHU94IeM5dp6N/wBrx4YbYu/k879WySJGOMsgB27oxNX/AB30/S4cYw44449Iivj76PPtvhz8I9UnbJGsuV5cPD4EoYTHLKcSf7Mp0B7K6AvovPT08TnM+FdWrtzwyrjhjhjWOEbHTss95IH4gNjJJrOJFXOUeVyeV6jgkbhDb1uMDKI+MtPxfUIEgQQCDoQehDx339XW9Jj2mXZuTWsPmuTy58rackYbh+1EEEjuOtMeVCOLPkhD7YyIGt/K3ybb3frjPmbyTayOsmCdFKcigkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkAJACQAkALqgBslx80BcsWSI7zE/uS3XadZfoJmeat6eJx8fJsSyjGey61Hw/xY3prNvHCpbjwcz649I8KcZ2M0ALMCCDI100sU4d/s4uf8p5Kx3/J5D9Rhw4oAShihjPtGj+It4PXrrJ0knybcrb5vmhhynpjmf8Axl+5+rfLi+V+j2OuXF8v+q8j/wBnL/2S/c/UPk7Nv5b9Hrdczzcn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Pmnpb3w+r1OndHJ4n+S5qH83HfdUtPm+2+f7G3nHodO+ObyY+iD9rMf/ABh+8vrPGf8AH89vwdm+/wBmHBH0jiir3y+Mqv6APe8/sae/1dGu+sufHweLikJRxREgbBJJIPzJehzPT0nSNL3XzQSAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIB5nO9M8W8uG9/UxJsS+BPQ+3R9N5ep6Oedevk6t67ebD5IgxJBBBBog9QX6fJxcGWW6eKEpd5H5974nrumt5sldnLNfOYONl5G7w47tos/uHv7P1EYxgKiBEdwAA/B8uut26PZ0dLZHJ43C9ONjLk2yEbPhdT00s9OvYX2nh6fpeNxceDu3tt4fiw8jnznzRGMIiMYeY7iN19Okd2j67w9S31Ok6fvwd29fysPk5454/vjKN9NwIv4W/VkA9QD8Rb47LOsw9js4vkn6s4sZ6wh/2x/c+J7MTyn0dnHL5R+q/V8J/9LH/ANkf3Pjezt18p9I7OWb5vlX6n9WwH/0cX/ZH9z43s7Nf5Z9I6uWb518s/ST9O4mTriA/0kx/I0+N6r6Wl8Po6uXdfN82+/8A5RxO7J/3/wCD5Xp+zp7/AFdXPvrwH6D/ACjid0/+8vmen7Onv9XRz767k9BkEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAEgBIASAf/9k=",Hh=`
  precision highp float;
  uniform float uTime;
  uniform float uBaseSize;
  attribute float aLuminance;
  attribute float aSize;
  attribute float aPhase;
  attribute float aCharIndex;
  varying float vFacing;
  varying float vLuminance;
  varying float vShimmer;
  varying vec3 vPosition;

  void main() {
    vPosition = position;
    vec3 viewNormal = normalize(normalMatrix * normalize(position));
    vFacing = dot(viewNormal, vec3(0.0, 0.0, 1.0));
    vLuminance = aLuminance;

    vec3 n = normalize(position);

    float wave1 = sin(uTime * 0.4 + n.x * 4.0 + n.y * 3.0) * 0.5 + 0.5;
    float wave2 = sin(uTime * 0.3 - n.z * 5.0 + n.y * 2.0) * 0.5 + 0.5;

    float bloom1 = pow(sin(uTime * 0.6 + n.x * 3.0 + n.z * 2.0) * 0.5 + 0.5, 4.0);
    float bloom2 = pow(sin(uTime * 0.45 - n.y * 4.0 + n.x * 1.5) * 0.5 + 0.5, 4.0);
    float bloom3 = pow(sin(uTime * 0.7 + n.z * 3.5 - n.x * 2.5) * 0.5 + 0.5, 4.0);
    float bloom4 = pow(sin(uTime * 0.35 + n.y * 2.5 + n.z * 3.0) * 0.5 + 0.5, 4.0);
    float blooms = max(bloom1, max(bloom2, max(bloom3, bloom4)));

    float wave = wave1 * 0.3 + wave2 * 0.2;
    wave = smoothstep(0.15, 0.7, wave);

    float landMask = smoothstep(0.05, 0.25, aLuminance);
    float cityBoost = smoothstep(0.3, 0.7, aLuminance) * blooms;
    float hotspot = blooms * landMask + cityBoost * 0.5;

    vShimmer = wave * 0.3 + hotspot * 0.7;
    float pulse = 0.95 + hotspot * 0.2;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = min(uBaseSize * aSize * pulse * (200.0 / -mvPosition.z), 64.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`,Qh=`
  precision highp float;
  uniform float uTime;
  uniform sampler2D uCircleTex;
  varying float vFacing;
  varying float vLuminance;
  varying float vShimmer;
  varying vec3 vPosition;

  void main() {
    float edge = step(0.0, vFacing) * smoothstep(0.0, 0.4, vFacing);

    vec4 circle = texture2D(uCircleTex, gl_PointCoord);

    float isLand = step(0.03, vLuminance);

    vec3 nn = normalize(vPosition);
    float warp1 = sin(nn.y * 3.0 + uTime * 0.12) * 0.5;
    float warp2 = cos(nn.x * 2.5 + uTime * 0.09) * 0.4;

    float spread1 = sin(nn.x * 2.0 + nn.z * warp1 + uTime * 0.18) *
                    sin(nn.z * 1.8 - nn.y * warp2 + uTime * 0.13);
    float spread2 = sin(nn.y * 2.5 + nn.x * warp2 + uTime * 0.14 + warp1) *
                    sin(nn.x * 1.5 + nn.z * warp1 - uTime * 0.1);
    float spread3 = sin(nn.z * 3.0 + nn.y * warp1 * 1.5 + uTime * 0.2) *
                    sin(nn.y * 2.0 - nn.x * warp2 * 1.2 + uTime * 0.16);

    float spread = spread1 * 0.4 + spread2 * 0.35 + spread3 * 0.25;

    float p = vLuminance * 137.0;
    float tick = sin(uTime * 0.3 + p) * 0.5 + 0.5;
    float dotVariance = 0.6 + spread * 0.3 + tick * 0.1;

    float basePulse = sin(uTime * 0.25) * 0.5 + 0.5;
    float base = mix(0.3, 0.52 + vLuminance * 0.15, isLand);
    base *= (0.8 + basePulse * 0.2) * dotVariance;

    float glow = vShimmer * 0.7;
    float alpha = circle.a * (base + glow) * edge;
    if (alpha < 0.005) discard;

    float brightness = mix(0.2, 0.4, isLand) * (0.8 + basePulse * 0.2) * dotVariance + vShimmer * 0.7;
    gl_FragColor = vec4(vec3(brightness), alpha);
  }
`,Nh=`
  varying vec3 vNormal;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`,Gh=`
  varying vec3 vNormal;
  void main() {
    float f = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 4.0);
    gl_FragColor = vec4(vec3(0.3), f * 0.02);
  }
`;function zh(t,e,n,i,r){const s=(r+180)/360,o=(90-i)/180,a=Math.floor(s*e)%e,l=(Math.floor(o*n)%n*e+a)*4;return(.2126*t[l]+.7152*t[l+1]+.0722*t[l+2])/255}function jh(t){const e=document.createElement("canvas");e.width=64*t.length,e.height=64;const n=e.getContext("2d");n.clearRect(0,0,e.width,64),n.font="48px 'SF Mono', 'Menlo', 'Consolas', monospace",n.textAlign="center",n.textBaseline="middle",n.fillStyle="#ffffff";for(let r=0;r<t.length;r++)n.fillText(t[r],r*64+32,32);const i=new hh(e);return i.flipY=!0,i.needsUpdate=!0,{texture:i,count:t.length}}function Uh(t,e,n,i,r,s,o){const a=[],A=[],l=[],c=[],d=[];for(let f=-85;f<=85;f+=i){const p=i/Math.cos(f*Math.PI/180);for(let E=-180;E<180;E+=p){const g=zh(t,e,n,f,E),h=(90-f)*Math.PI/180,u=(E+180)*Math.PI/180;a.push(-s*Math.sin(h)*Math.cos(u),s*Math.cos(h),s*Math.sin(h)*Math.sin(u)),A.push(g),l.push(1),c.push(Math.random()*Math.PI*2),d.push(Math.floor(Math.random()*o))}}return{positions:a,luminances:A,sizes:l,phases:c,charIndices:d}}function xo(t={}){const{container:e,gridStep:n=.9,dotSize:i=2,radius:r=7,tilt:s=[35,-23.5],rotationSpeed:o=6e-4,cameraDistance:a=18,luminanceThreshold:A=.08,chars:l=".",backgroundColor:c=0,backgroundOpacity:d=1,atmosphere:f=!0,atmosphereOpacity:p=.02,timeOffset:E=12,nightImageUrl:g=Dh}=t;let h;typeof e=="string"?h=document.querySelector(e):e instanceof HTMLElement?h=e:(h=document.querySelector("[data-dot-globe]"),h||(h=document.createElement("div"),h.style.cssText="position:fixed;inset:0;z-index:-1;",document.body.prepend(h)));const u=document.createElement("canvas");u.style.cssText="width:100%;height:100%;display:block;",h.appendChild(u);const v=h.getBoundingClientRect(),y=v.width||window.innerWidth,T=v.height||window.innerHeight,_=new uh;d<1?_.background=null:_.background=new Ne(c);const C=new Pt(60,y/T,.1,1e3);C.position.set(0,0,a),C.lookAt(0,0,0);const B=d<1,z=new mo({canvas:u,antialias:!0,alpha:B});B&&z.setClearColor(new Ne(c),d),z.setSize(y,T),z.setPixelRatio(Math.min(window.devicePixelRatio,2));const{texture:M,count:I}=jh(l),H=new Image;H.crossOrigin="anonymous",H.src=g;let W=null,Y=null,b=null;H.onload=()=>{const j=document.createElement("canvas");j.width=H.width,j.height=H.height;const X=j.getContext("2d");X.drawImage(H,0,0);const ee=X.getImageData(0,0,H.width,H.height).data,k=Uh(ee,H.width,H.height,n,A,r,I);b=new un,b.setAttribute("position",new mt(k.positions,3)),b.setAttribute("aLuminance",new mt(k.luminances,1)),b.setAttribute("aSize",new mt(k.sizes,1)),b.setAttribute("aPhase",new mt(k.phases,1)),b.setAttribute("aCharIndex",new mt(k.charIndices,1)),Y=new Vt({uniforms:{uTime:{value:0},uBaseSize:{value:i},uCharTex:{value:M},uCharCount:{value:I}},vertexShader:Hh,fragmentShader:Qh,transparent:!0,blending:dn,depthWrite:!1});const Z=new fh(b,Y);if(Z.rotation.y=kr.degToRad(-230),W=new pi,W.rotation.x=kr.degToRad(s[0]),W.rotation.z=kr.degToRad(s[1]),W.add(Z),f){const re=Gh.replace("f * 0.02",`f * ${p.toFixed(4)}`),se=new ph(r*1.12,64,64),me=new Vt({vertexShader:Nh,fragmentShader:re,transparent:!0,side:ut,depthWrite:!1});W.add(new Yt(se,me))}_.add(W)};const Q=new Ph;let G;const q=()=>{G=requestAnimationFrame(q);const j=Q.getDelta(),X=Q.elapsedTime+E;W&&W.children.length>0&&(W.children[0].rotation.y+=o*j*60),Y&&(Y.uniforms.uTime.value=X),z.render(_,C)};q();const U=()=>{const j=h.getBoundingClientRect(),X=j.width||window.innerWidth,ee=j.height||window.innerHeight;C.aspect=X/ee,C.updateProjectionMatrix(),z.setSize(X,ee)};return window.addEventListener("resize",U),{destroy(){window.removeEventListener("resize",U),cancelAnimationFrame(G),b?.dispose(),Y?.dispose(),M.dispose(),z.dispose(),u.remove()}}}function qh(){if(typeof window>"u")return{};const t=new URLSearchParams(window.location.search),e={};return t.has("gridStep")&&(e.gridStep=parseFloat(t.get("gridStep"))),t.has("dotSize")&&(e.dotSize=parseFloat(t.get("dotSize"))),t.has("radius")&&(e.radius=parseFloat(t.get("radius"))),t.has("rotationSpeed")&&(e.rotationSpeed=parseFloat(t.get("rotationSpeed"))),t.has("backgroundColor")&&(e.backgroundColor=parseInt(t.get("backgroundColor"),16)),t.has("backgroundOpacity")&&(e.backgroundOpacity=parseFloat(t.get("backgroundOpacity"))),t.has("atmosphere")&&(e.atmosphere=t.get("atmosphere")!=="false"),(t.has("tiltX")||t.has("tiltZ"))&&(e.tilt=[parseFloat(t.get("tiltX")||"35"),parseFloat(t.get("tiltZ")||"-23.5")]),e}return typeof window<"u"&&(window.DotGlobe={create:xo}),Go(ds)})();/*! Bundled license information:

three/build/three.module.js:
  (**
   * @license
   * Copyright 2010-2023 Three.js Authors
   * SPDX-License-Identifier: MIT
   *)
*/
