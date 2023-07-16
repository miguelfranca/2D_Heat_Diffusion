#include "Application.h"
#include <algorithm>

#define UP_SCALE 3.0
#define MAX_HEAT_RADIUS 60.f // pixels
#define OUTPUT_EVERY 10

Application::Application(std::string t)
{
	title = t;
	nx = 300;
	ny = 300;

	heat_radius = 10.;

	h_O = new float[nx * ny];
	h_barriers = new bool[nx * ny];

	pixels = new sf::Uint8[nx * ny];
	texture.create(nx, ny);
	sprite.setPosition(0, 0);
	image.create(nx, ny);

	setupWindow(nx * UP_SCALE, ny * UP_SCALE);
}

// called once before the loop starts
bool Application::onCreate()
{
	// prepare GPU
	prepare(h_O, h_barriers, nx, ny);

	sprite.setTexture(texture, true);

	setExitKey(sf::Keyboard::Escape);
	return true;
}

// first thing to be called every frame
bool Application::onHandleEvent(GF::Event& event)
{

	// add heat
	if (GF::Mouse::Left.isPressed()) {
		if (GF::Mouse::isInsideWindow(window)) {
			sf::Vector2f pos = GF::Mouse::getPosition(window);
			addHeat(pos.x / UP_SCALE, pos.y / UP_SCALE, 100.0, heat_radius / UP_SCALE);
		}
	}

	// add cold
	if (GF::Mouse::Right.isPressed()) {
		if (GF::Mouse::isInsideWindow(window)) {
			sf::Vector2f pos = GF::Mouse::getPosition(window);
			addHeat(pos.x / UP_SCALE, pos.y / UP_SCALE, -100.0, heat_radius / UP_SCALE);
		}
	}

	if(GF::Mouse::Wheel.isPressed()){
		if(GF::Mouse::isInsideWindow(window)){
			sf::Vector2f pos = GF::Mouse::getPosition(window);
			addBarrier(pos.x / UP_SCALE, pos.y / UP_SCALE, heat_radius / UP_SCALE);
		}
	}
	if(event.isNothing()) return true;

	// change brush size
	if (GF::Mouse::Wheel.moved(event)) {
		heat_radius += GF::Mouse::Wheel.delta(event);
		heat_radius = std::max(1.f, std::min(MAX_HEAT_RADIUS, heat_radius));
	}

	// handle waiting events
	int c = 0;
	while (window.pollEvent(event) && c++ < 5)
		onHandleEvent(event);

	return true;
}

// from https://stackoverflow.com/a/40639903
// input: ratio is between 0.0 to 1.0
// output: rgb color
sf::Color rgb(float ratio)
{
    // we want to normalize ratio so that it fits into 6 regions
    // where each region is 256 units long
    int normalized = int(ratio * 256 * 6);

    // find the region for this position
    int region = normalized / 256;

    // find the distance to the start of the closest region
    int x = normalized % 256;

    uint8_t r = 0, g = 0, b = 0;
    switch (region)
    {
	    case 0: r = 255; g = 0;   b = 0;   g += x; break;
	    case 1: r = 255; g = 255; b = 0;   r -= x; break;
	    case 2: r = 0;   g = 255; b = 0;   b += x; break;
	    case 3: r = 0;   g = 255; b = 255; g -= x; break;
	    case 4: r = 0;   g = 0;   b = 255; r += x; break;
	    case 5: r = 255; g = 0;   b = 255; b -= x; break;
    }
    return sf::Color(r, g, b);
}

// called every frame before draw
bool Application::onUpdate(const float fElapsedTime, const float fTotalTime)
{
	static int step = 0;

	launchKernel(step++, OUTPUT_EVERY, h_O, h_barriers);

	if (step % OUTPUT_EVERY == 0) {
		for (int y = 0; y < ny; ++y) {
			for (int x = 0; x < nx; ++x){
				float value = std::min(1.0, std::max(-1.0, h_O[y * nx + x] / 30.0));
				bool is_barrier = h_barriers[y * nx + x];

				if(is_barrier)
					image.setPixel(x, y, GRAY);
				else {
					sf::Color color = rgb((value + 1.0) / 2. * 2. / 3.);
					// sf::Color color = rgb((1. - value)*(1. - value) * 2. / 3.);
					image.setPixel(x, y, color);
				}
			}
		}

		texture.loadFromImage(image);
		texture.setSmooth(true);
		sprite.setTexture(texture);
		sprite.setScale(UP_SCALE, UP_SCALE);
	}

	return true;
}

// last thing to be called every frame
bool Application::onDraw()
{
	window.draw(sprite);
	window.draw(GF::Circle(heat_radius, GF::Mouse::getPosition(window), TRANSPARENT, BLACK, 1.0));

	return true;
}

// called before exiting the app
void Application::onDestroy()
{
	finalize();
	delete [] pixels;
	delete [] h_O;
}

void Application::onSwitch(std::string other)
{

}
