#include "Application.h"

int main()
{
	Application app;
	app.showFPS(true);
	app.setMaxFPS(500);
	app.run();

	return 0;
}
