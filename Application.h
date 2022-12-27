#pragma once

// #include "GraphicsFramework/GUI.h"
// #include "GraphicsFramework/Tools.h"
#include "GraphicsFramework/main.h"
// #include "GraphicsFramework/sfml.h"
#include "evolve.h"

class Application : public GF::App
{
public:
    Application(std::string t = "Window");

    // called once before the loop starts
    bool onCreate() override;

    // first thing to be called every frame
    bool onHandleEvent(GF::Event& event) override;

    // called every frame before draw
    virtual bool onUpdate(const float fElapsedTime, const float fTotalTime) override;

    // last thing to be called every frame
    bool onDraw() override;

    // called once before exiting the app
    void onDestroy() override;

    void onSwitch(std::string other) override;

private:
    int nx, ny;
    
    float* h_O;
    sf::Uint8* pixels;
    sf::Texture texture;
    sf::Sprite sprite;
    sf::Image image;

    float heat_radius;
};
