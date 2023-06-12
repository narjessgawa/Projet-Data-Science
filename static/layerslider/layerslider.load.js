$(document).ready(function () {
	"use strict";

	// Calling LayerSlider 
		
        $('#layerslider').layerSlider({
            responsive: true,
            responsiveUnder: 1280,
            layersContainer: 1280,
            skin: 'fullwidth',
            hoverPrevNext: false,
            skinsPath: './layerslider/skins/',
            autoStart: true,
			autoPlayVideos : false
        });
		
		
 });