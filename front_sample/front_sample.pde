import http.requests.*;
Brain brain;
void setup() {
  size(500, 500);
  brain = new Brain("http://localhost", "8000");
  background(0);
  cx = width / 2;
  cy = height / 2;
}

void draw() {
  if (!stop) {
    background(0);
    show_now_and_future();
  }
}

float[] split_params(String s) {
  return float(split(s.replace('"', ' ').trim(), ','));
}

PVector params2vector(String s) {
  float[] p = split_params(s);
  float x = scale_up(p[0], 0, width);
  float y = scale_up(p[1], 0, height);
  return new PVector(x, y);
}

float scale_down(int i, int min, int max) {
  return map(i, min, max, -1, 1);
}

float scale_up(float i, int min, int max) {
  return map(i, -1, 1, min, max);
}

String join_params(float x, float y) {
  return x + "," + y;
}

String join_samples(String a, String b) {
  return a + ":" + b;
}

String joined_sample(int _x, int _y, int _px, int _py) {
  float x = scale_down(_x, 0, width);
  float y = scale_down(_y, 0, height);
  float px = scale_down(_px, 0, width);
  float py = scale_down(_py, 0, height);
  String now = join_params(x, y);
  String past = join_params(px, py);
  return join_samples(now, past);
}

String joined_sample(int _x, int _y) {
  float x = scale_down(_x, 0, width);
  float y = scale_down(_y, 0, height);
  String now = join_params(x, y);
  String past = "";
  return join_samples(now, past);
}

ArrayList<PVector> ps;
int xx, yy;
float cx, cy;
boolean auto = true;
boolean predict_mode = true;
boolean stop = false;
int px, py;

void show_desc() {
  fill(255, 0, 0);
  text("Current Position", 100, 100);
  fill(0, 255, 0);
  text("Precited", 100, 200);
}

void show_now_and_future() {
  fill(255, 0, 0, 200);
  ellipse(xx, yy, 10, 10);
  if (auto) {
    xx = int(cx + width / 3 * cos(frameCount / 20.0));
    yy = int(cy + height / 3 * sin(frameCount / 20.0));
  } else {
    xx = mouseX;
    yy = mouseY;
  }
  ArrayList<PVector> ps = brain.predict(xx, yy, 4);
  for (int i = 0; i < ps.size (); i++) {
    PVector p = ps.get(i);
    fill(0, 255, 0, 200 - i * 2);
    stroke(0, 255, 0, 200);
    noStroke();
    ellipse(p.x, p.y, 6, 6);
  }
  brain.learn(px, py, xx, yy);
  px = xx;
  py = yy;
}

class Brain {
  PostRequest post;
  Brain(String host, String port) {
    post = new PostRequest(host + ":" + port);
  }
  void learn(int x, int y, int px, int py) {
    post.addData("q", joined_sample(x, y, px, py));
    post.send();
  }
  ArrayList<PVector> predict(int x, int y, int future) {
    ArrayList<PVector> ps = new ArrayList<PVector>();
    for (int i = 0; i < future; i++) {
      post.addData("q", joined_sample(x, y));
      post.send();
      String params = post.getContent();
      PVector p = params2vector(params);
      ps.add(p);
      x = (int)p.x;
      y = (int)p.y;
    }
    return ps;
  }
}

void keyPressed() {
  stop = !stop;
}

