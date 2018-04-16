#ifndef __SDL2SHOW_HPP_
#define __SDL2SHOW_HPP_
//note! not thread-safe!!!
#define SDLSHOW_MAXID 50
#if defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <SDL2/SDL.h>

class SDL2Show {
public:
	static SDL2Show& Instance() {
			static SDL2Show theSDL2Show;
			return theSDL2Show;
	}
	/* more (non-static) functions here */
	void imshow(int id,cv::Mat const &img);

private:
	typedef struct __winptr {
		SDL_Window *pw;
		SDL_Renderer *pr;
		SDL_Texture *pt;
		cv::Size wh;
	}winptr;
	SDL2Show();
	SDL2Show(SDL2Show const&){}              // copy ctor hidden
	SDL2Show& operator=(SDL2Show const&){}   // assign op. hidden
	~SDL2Show();
	void destory(int id);
	winptr _winptr[SDLSHOW_MAXID];
	bool binit;
};

#endif //USE_OPENCV
#endif //__SDL2SHOW_HPP_