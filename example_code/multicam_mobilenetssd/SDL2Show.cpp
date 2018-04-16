#include "SDL2Show.hpp"
#include "detector.hpp"

#if defined(USE_OPENCV)
//Section "Device"
//       Identifier      "Configured Video Device"
//       Driver          "intel"   ==> "fbdev"
//EndSection
//https://www.linuxtv.org/downloads/legacy/video4linux/API/V4L2_API/spec/ch04s04.html
//-----
//sudo apt-get install sudo apt-get install libdirectfb-dev 
//./configure --enable-video-directfb  when compile sdl2
//?https://searchcode.com/codesearch/view/69047742/
//export SDL_VIDEODRIVER=directfb still slow
//????? SDL  -- XCB  (vlc change settings to all to see it's xcb, XVideo output (XCB)) -- https://github.com/videolan/vlc/blob/a9712a7e4fba854fceab8ce5f232b6baf05dc39c/modules/video_output/xcb/xvideo.c
//apt-get install libxcb-composite0-dev libxcb-glx0-dev libxcb-dri2-0-dev libxcb-xf86dri0-dev libxcb-xinerama0-dev libxcb-render-util0-dev
//xvinfo to check 
//https://github.com/ShabbyX/vktut/blob/c9d186fe12887addbb2133a9f50959fe93af87dc/tut6/tut6.c
SDL2Show::SDL2Show(){
	memset(_winptr,0,sizeof(_winptr));
	binit = true;
	if(SDL_Init(SDL_INIT_VIDEO)!=0) {
		std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
		binit = false;
	}
}        
	
SDL2Show::~SDL2Show(){
	for(int i=0;i<SDLSHOW_MAXID;i++){
		destory(i);
	}
	SDL_Quit();
}

void SDL2Show::destory(int id){
		if(_winptr[id].pt!=NULL)
			SDL_DestroyTexture(_winptr[id].pt);
		if(_winptr[id].pr!=NULL)
			SDL_DestroyRenderer(_winptr[id].pr);
		if(_winptr[id].pw!=NULL)
			SDL_DestroyWindow(_winptr[id].pw);	
		memset(&_winptr[id],0,sizeof(winptr));
}

void SDL2Show::imshow(int id,cv::Mat const &img) {
	if(id>=SDLSHOW_MAXID||!binit)
		return;
	if(_winptr[id].wh!=img.size())
		destory(id);
	if(_winptr[id].pt==NULL){
		_winptr[id].wh = img.size();
		std::string title = "input"+std::to_string(id);
		_winptr[id].pw=SDL_CreateWindow(title.c_str(),640*(id%3),(id>2)?480:0, _winptr[id].wh.width, _winptr[id].wh.height, 0);//SDL_WINDOW_OPENGL);
		_winptr[id].pr=SDL_CreateRenderer(_winptr[id].pw, -1,  SDL_RENDERER_ACCELERATED);
    _winptr[id].pt = SDL_CreateTexture(
      _winptr[id].pr,
      SDL_PIXELFORMAT_BGR24,
      SDL_TEXTUREACCESS_STREAMING,
      _winptr[id].wh.width,
      _winptr[id].wh.height
    );		
	}
	//display data
	SDL_UpdateTexture( _winptr[id].pt, NULL, img.data, _winptr[id].wh.width*3);
	SDL_Event event;
	SDL_PollEvent(&event);
	SDL_RenderClear(_winptr[id].pr);
	SDL_RenderCopy(_winptr[id].pr, _winptr[id].pt, NULL, NULL);
	SDL_RenderPresent(_winptr[id].pr);
}

#endif //USE_OPENCV