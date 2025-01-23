// JavaScript to send the request

const imagesDir = 'static/images/';
const imageClass = 'imgBox';

function loadFileNames() {
	$.ajax({
		url: '/files',
		type: 'GET',
		success: function(files) {
			let file_labels = files.labels;
			// console.log(file_labels);
			files = files.files;
			// console.log(files);
			let totals = files.length;
			let box_img = '';
			let leg_item = '';
			$('.navBar li p span.totals').text(totals);
			let uniq = [];
			let legend_labels = file_labels.filter(obj => !uniq[obj] && (uniq[obj] = true)).flat(1);
			console.log(legend_labels);

			$.each(legend_labels, function(i, v) {
				leg_item += '<li>';
				leg_item += '<i class="mdi mdi-circle ' + v + '"></i>';
				leg_item += '<span>' + v + '</span>';
				leg_item += '</li>';
			});

			$('.legendBox').append(leg_item);

			$.each(files, function(i, v) {
				// console.log(file_labels[i]);
				let imgSrc = imagesDir + v;
				let altImg = v.replace(/\.[^/.]+$/, '');
				box_img += '<figure class=' + file_labels[i] + '>';
				box_img += '<figcaption>' + altImg + '</figcaption>';
				box_img += '<i class="imgBox mdi mdi-file" id=' + i + ' data-img=' + altImg + ' ></i>';
				/*box_img += '<img id=' + i + ' class=' + imageClass + ' src=' + imgSrc + ' alt=' + altImg + ' />';*/
				box_img += '</figure>';
			});

			$('.gridImg').append(box_img);

			setTimeout(function() {
				$('.imgBox').on('click', function() {
					// let thisAttr = $(this).attr('alt');
					let thisAttr = $(this).attr('data-img');
					$('.gridImg figure').removeClass('selected');
					$(this).parent().addClass('selected');
					getPrediction(thisAttr);
				});
			}, 250);
		},
		error: function(err) {
			console.log(err);
		}
	});
}

function getPrediction(attr) {
	$('.ecgBox').remove();
	$('.mainStage').addClass('colGap');
	$('.viewBox').removeClass('dispNone').spin();
	$('.gridImg').removeClass('fullGrid').addClass('minGrid');
	$.ajax({
		url: '/predict',
		type: 'POST',
		data: JSON.stringify({
			'ecg_data': attr
		}),
		success: function(res) {
			$('.spinner').remove();
			console.log(res);
			let imgSrc = imagesDir + res.ecg_data + '.png';

			let bubble = '';
			bubble += '<div class="ecgBox">';
			bubble += '<i class="closeBtn mdi mdi-close"></i>';
			bubble += '<img src=' + imgSrc + ' alt=' + res.ecg_data + ' />';
			bubble += '<p>' + res.prediction + '</p>';
			bubble += '<p>' + res.real + '</p>';
			bubble += '</div>';

			setTimeout(function() {
				$('.viewBox').removeClass('dispNone').append(bubble);
				$('.ecgBox').hide().fadeIn();
				$('.closeBtn').on('click', function() {
					$('.viewBox').addClass('dispNone');
					$('.ecgBox').remove();
					$('.gridImg').removeClass('minGrid').addClass('fullGrid');
					$('.mainStage').removeClass('colGap');
					$('.gridImg figure').removeClass('selected');
				});
			}, 250);
		},
		error: function(err) {
			console.log(err);
		},
		dataType: 'json',
		contentType: 'application/json'
	});

}

function changeModel() {
	$('#selModel').change(function() {
		let model_val = $(this).val();

		$.ajax({
			url: '/change_model',
			type: 'POST',
			data: JSON.stringify({
				'model': model_val
			}),
			success: function(res) {
				console.log(model_val);
			},
			error: function(err) {
				console.log(err);
			},
			dataType: 'json',
			contentType: 'application/json'
		});
	});

}

$(window).on('load', function() {
	loadFileNames();
});

$(window).resize(function() {
});

$(document).ready(function() {
	changeModel();
});
