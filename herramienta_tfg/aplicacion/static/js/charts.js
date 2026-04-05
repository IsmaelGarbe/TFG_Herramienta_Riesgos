function pintarCurvas(curvas){
  if (typeof Chart === 'undefined') return;

  const ctx1 = document.getElementById('lossMagnitudeChart')?.getContext('2d');
  if (ctx1) new Chart(ctx1, {
    type: 'line',
    data: {
      labels: curvas.lossMagnitude.labels,
      datasets: [{
        label:'Loss Magnitude/yr',
        data:curvas.lossMagnitude.probs,
        borderWidth:2,
        fill:false
      }]
    },
    options: {
      responsive:true,
      scales:{
        x:{
          title:{display:true,text:'€'},
          ticks: {
            callback: function(value, index) {
              const realValue = this.chart.data.labels[index];
              return new Intl.NumberFormat('es-ES').format(realValue);
            }
          }
        },
        y:{
          title:{display:true,text:'%'}
        }
      }
    }
  });

  const ctx2 = document.getElementById('exceedanceChart')?.getContext('2d');
  if (ctx2) new Chart(ctx2, {
    type: 'line',
    data: {
      labels: curvas.exceedance.labels,
      datasets: [{
        label:'Chance of Exceeding',
        data:curvas.exceedance.probs,
        borderWidth:2,
        fill:false
      }]
    },
    options: {
      responsive:true,
      scales:{
        x:{
          title:{display:true,text:'€'},
          ticks: {
            callback: function(value, index) {
              const realValue = this.chart.data.labels[index];
              return new Intl.NumberFormat('es-ES').format(realValue);
            }
          }
        },
        y:{
          title:{display:true,text:'% Probabilidad'}
        }
      }
    }
  });
}

window.Charts = { pintarCurvas };