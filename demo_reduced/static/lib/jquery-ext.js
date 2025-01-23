(function($) {
	$.fn.serializeObject = function() {
		let o = {};
		let a = this.serializeArray();
		$.each(a, function() {
			if (o[this.name]) {
				if (!o[this.name].push) {
					o[this.name] = [o[this.name]];
				}
				o[this.name].push(this.value || '');
			}
			else {
				o[this.name] = this.value || '';
			}
		});
		return o;
	};

	$.fn.toggleAttr = function(attr, attr1, attr2) {
		return this.each(function() {
			let self = $(this);
			if (self.attr(attr) == attr1)
				self.attr(attr, attr2);
			else
				self.attr(attr, attr1);
		});
	};

	$.fn.toggleInputValue = function() {
		return $(this).each(function() {
			let input = $(this);
			let defaultValue = input.val();

			input.focus(function() {
				alert('no');
				if (input.val() == defaultValue)
					input.val('');
			}).blur(function() {
				if (input.val().length == 0)
					input.val(defaultValue);
			});
		});
	};

	$.fn.spin = function(opts) {
		let preset = {
			lines: 13, // The number of lines to draw
			length: 5, // The length of each line
			width: 2, // The line thickness
			radius: 5, // The radius of the inner circle
			corners: 1, // Corner roundness (0..1)
			rotate: 0, // The rotation offset
			direction: 1, // 1: clockwise, -1: counterclockwise
			color: '#616264', // #rgb or #rrggbb
			speed: 1, // Rounds per second
			trail: 50, // Afterglow percentage
			shadow: false, // Whether to render a shadow
			hwaccel: false, // Whether to use hardware acceleration
			className: 'spinner', // The CSS class to assign to the spinner
			zIndex: 2e9, // The z-index (defaults to 2000000000)
			top: 'auto', // Top position relative to parent in px
			left: 'auto' // Left position relative to parent in px
		};

		let data = $(this).data();

		if (data.spinner) {
			data.spinner.stop();
			delete data.spinner;
		}

		if (opts != false) {
			data.spinner = new Spinner(preset).spin();

			$(this).append(data.spinner.el);
		}
	};

})(jQuery);
