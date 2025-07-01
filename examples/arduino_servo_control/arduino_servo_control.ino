#include <PCA9685.h>

#define THUMB_JOINT_1   0
#define THUMB_JOINT_2   1
#define INDEX_JOINT_1   2
#define INDEX_JOINT_2   3
#define MIDDLE_JOINT_1  4
#define MIDDLE_JOINT_2  5
#define RING_JOINT_1    6
#define RING_JOINT_2    7
#define PINKY_JOINT_1   8
#define PINKY_JOINT_2   9
#define WRIST_ROLL     10

#define SERVOMIN 150
#define SERVOMAX 600

PCA9685 pwm = PCA9685(0x40);
#define PCSerial Serial  // UnoなどではSerialを共用

void servo_write(uint8_t channel, uint8_t angle) {
  angle = constrain(angle, 0, 180);
  uint16_t pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(channel, 0, pulse);
}

void setup() {
  Serial.begin(115200);
  PCSerial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(50);

  for (int i = 0; i <= 10; i++) {
    servo_write(i, 90);
    delay(100);
  }

  delay(1000);
  Serial.println("初期化完了。'0'=開く, '1'=閉じる");
}

char lastCommand = '\0';  // グローバル変数として宣言

void loop() {
  if (PCSerial.available() > 0) {
    char receivedChar = PCSerial.read();

    // 前回と同じコマンドなら処理しないが、シリアルバッファはクリアする
    if (receivedChar == lastCommand) {
      while (PCSerial.available() > 0) PCSerial.read();
      return;  // ここはreturnでも問題ないですが、処理は終わり
    }

    if (receivedChar == '1') {
      Serial.println("全関節を開きます");
      for (int pos = 90; pos >= 0; pos -= 5) {
        for (int joint = 0; joint <= 10; joint++) {
          servo_write(joint, pos);
        }
        delay(30);
      }
      Serial.println("Hand Open 完了");

    } else if (receivedChar == '0') {
      Serial.println("全関節を閉じます");
      for (int pos = 0; pos <= 90; pos += 5) {
        for (int joint = 0; joint <= 10; joint++) {
          servo_write(joint, pos);
        }
        delay(30);
      }
      Serial.println("Hand Close 完了");

    } else {
      Serial.print("無効な入力です: ");
      Serial.println(receivedChar);
    }

    lastCommand = receivedChar;
    while (PCSerial.available() > 0) PCSerial.read();
  }
}
