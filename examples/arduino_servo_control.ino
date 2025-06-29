
#include <Servo.h>

Servo myServo;  // サーボオブジェクトを作成

const int servoPin = 9; // サーボモーターを接続するピン

void setup() {
  Serial.begin(9600); // シリアル通信を9600bpsで開始
  myServo.attach(servoPin); // サーボモーターをピンにアタッチ
  myServo.write(0); // 初期状態としてサーボを0度に設定（開いている状態）
  Serial.println("Arduino Ready");
}

void loop() {
  if (Serial.available() > 0) { // シリアルデータが利用可能かチェック
    char receivedChar = Serial.read(); // 受信した文字を読み込む

    if (receivedChar == '0') { // '0' は手が開いている状態
      myServo.write(0); // サーボを0度に設定
      Serial.println("Hand Open (0 degrees)");
    } else if (receivedChar == '1') { // '1' は手が閉じている状態
      myServo.write(90); // サーボを90度に設定（例：閉じる動作）
      Serial.println("Hand Closed (90 degrees)");
    }
  }
}


