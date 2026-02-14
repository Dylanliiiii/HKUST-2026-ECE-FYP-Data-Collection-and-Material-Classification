/*
 * CNC Shield V3 - XYZ 位移执行器
 * D/U : Z轴下/上
 * X/x : X轴 + / -
 * Y/y : Y轴 + / -
 * S   : 全部停止
 */

#include <AccelStepper.h>

// ---- 引脚 ----
#define X_STEP_PIN 2
#define X_DIR_PIN  5
#define Y_STEP_PIN 3
#define Y_DIR_PIN  6
#define Z_STEP_PIN 4
#define Z_DIR_PIN  7
#define ENABLE_PIN 8

AccelStepper stepperX(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
AccelStepper stepperY(AccelStepper::DRIVER, Y_STEP_PIN, Y_DIR_PIN);
AccelStepper stepperZ(AccelStepper::DRIVER, Z_STEP_PIN, Z_DIR_PIN);

// ---- 参数 ----
const float STEPS_PER_MM = 1600.0;  // T8 + 1/16
const float STEP_MM = 1;         // 每条命令 ？ mm

long stepUnit() {
  return STEP_MM * STEPS_PER_MM;
}

void setup() {
  Serial.begin(115200);

  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);

  stepperX.setMaxSpeed(8000);
  stepperY.setMaxSpeed(8000);
  stepperZ.setMaxSpeed(1000);

  stepperX.setAcceleration(8000);
  stepperY.setAcceleration(8000);
  stepperZ.setAcceleration(1000);

  Serial.println("XYZ executor ready");
}

void loop() {
  stepperX.run();
  stepperY.run();
  stepperZ.run();

  if (Serial.available()) {
    char c = Serial.read();

    if (c == 'D') stepperZ.move(stepUnit());
    else if (c == 'U') stepperZ.move(-stepUnit());

    else if (c == 'X') stepperX.move(stepUnit());
    else if (c == 'x') stepperX.move(-stepUnit());

    else if (c == 'Y') stepperY.move(stepUnit());
    else if (c == 'y') stepperY.move(-stepUnit());

    else if (c == 'S') {
      stepperX.stop(); stepperY.stop(); stepperZ.stop();
    }
  }
}
