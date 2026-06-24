#pragma once

#include <cmath>

namespace vinylizer {

enum class Easing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Spring
};

class Transition {
public:
    float pos_x = 0, pos_y = 0;
    float opacity = 1.0f;

    bool active = false;
    float start_x = 0, start_y = 0, start_opacity = 1;
    float target_x = 0, target_y = 0, target_opacity = 1;
    float duration_ms = 300;
    float elapsed_ms = 0;
    Easing easing = Easing::EaseInOut;

    // Spring physics
    float spring_velocity = 0;
    float spring_stiffness = 300.0f;
    float spring_damping = 20.0f;

    void to(float tx, float ty, float to, float dur_ms, Easing e = Easing::EaseInOut) {
        start_x = pos_x; start_y = pos_y; start_opacity = opacity;
        target_x = tx; target_y = ty; target_opacity = to;
        duration_ms = dur_ms; elapsed_ms = 0;
        easing = e; active = true;
        spring_velocity = 0;
    }

    void slide_out_left(float content_w, float dur = 250) {
        to(-content_w, pos_y, 0.0f, dur, Easing::EaseIn);
    }

    void slide_in_from_right(float content_w, float dur = 250) {
        pos_x = content_w; opacity = 0;
        to(0, pos_y, 1.0f, dur, Easing::EaseOut);
    }

    void fade_out(float dur = 200) {
        to(pos_x, pos_y, 0.0f, dur, Easing::EaseInOut);
    }

    void fade_in(float dur = 200) {
        opacity = 0;
        to(pos_x, pos_y, 1.0f, dur, Easing::EaseInOut);
    }

    bool update(float dt_ms) {
        if (!active) return false;

        if (easing == Easing::Spring) {
            // Damped spring for pos_x only
            float force = spring_stiffness * (target_x - pos_x);
            float damping_force = spring_damping * spring_velocity;
            spring_velocity += (force - damping_force) * (dt_ms / 1000.0f);
            pos_x += spring_velocity * (dt_ms / 1000.0f);
            opacity += (target_opacity - opacity) * std::min(1.0f, dt_ms / 200.0f);

            float diff = std::abs(pos_x - target_x) + std::abs(spring_velocity);
            if (diff < 0.5f) {
                pos_x = target_x; opacity = target_opacity; active = false;
            }
            return active;
        }

        elapsed_ms += dt_ms;
        float t = std::min(1.0f, elapsed_ms / duration_ms);
        float et = apply_easing(t, easing);

        pos_x = start_x + (target_x - start_x) * et;
        pos_y = start_y + (target_y - start_y) * et;
        opacity = start_opacity + (target_opacity - start_opacity) * et;

        if (t >= 1.0f) {
            pos_x = target_x; pos_y = target_y; opacity = target_opacity;
            active = false;
        }
        return active;
    }

    static float apply_easing(float t, Easing e) {
        switch (e) {
        case Easing::Linear:   return t;
        case Easing::EaseIn:   return t * t;
        case Easing::EaseOut:  return 1.0f - (1.0f - t) * (1.0f - t);
        case Easing::EaseInOut:
            return t < 0.5f ? 2 * t * t : 1 - (-2 * t + 2) * (-2 * t + 2) / 2;
        case Easing::Spring:   return t; // handled separately
        }
        return t;
    }
};

} // namespace vinylizer
