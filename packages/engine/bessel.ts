/*
Cephes I0 coefficients and Clenshaw recurrence, adapted from
https://github.com/scipy/xsf/blob/main/include/xsf/cephes/i0.h
https://github.com/scipy/xsf/blob/main/include/xsf/cephes/chbevl.h

Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1985, 1987, 1995, 2000 by Stephen L. Moshier

This software is derived from the Cephes Math Library and is
incorporated herein by permission of the author.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

'use strict';

const SMALL = [
  -4.4153416464793395e-18,
  3.3307945188222384e-17,
  -2.431279846547955e-16,
  1.715391285555133e-15,
  -1.1685332877993451e-14,
  7.676185498604936e-14,
  -4.856446783111929e-13,
  2.95505266312964e-12,
  -1.726826291441556e-11,
  9.675809035373237e-11,
  -5.189795601635263e-10,
  2.6598237246823866e-9,
  -1.300025009986248e-8,
  6.046995022541919e-8,
  -2.670793853940612e-7,
  0.0000011173875391201037,
  -0.0000044167383584587505,
  0.000016448448070728896,
  -0.00005754195010082104,
  0.00018850288509584165,
  -0.0005763755745385824,
  0.0016394756169413357,
  -0.004324309995050576,
  0.010546460394594998,
  -0.02373741480589947,
  0.04930528423967071,
  -0.09490109704804764,
  0.17162090152220877,
  -0.3046826723431984,
  0.6767952744094761,
];

const LARGE = [
  -7.233180487874754e-18,
  -4.830504485944182e-18,
  4.46562142029676e-17,
  3.461222867697461e-17,
  -2.8276239805165836e-16,
  -3.425485619677219e-16,
  1.7725601330565263e-15,
  3.8116806693526224e-15,
  -9.554846698828307e-15,
  -4.150569347287222e-14,
  1.54008621752141e-14,
  3.8527783827421426e-13,
  7.180124451383666e-13,
  -1.7941785315068062e-12,
  -1.3215811840447713e-11,
  -3.1499165279632416e-11,
  1.1889147107846439e-11,
  4.94060238822497e-10,
  3.3962320257083865e-9,
  2.266668990498178e-8,
  2.0489185894690638e-7,
  0.0000028913705208347567,
  0.00006889758346916825,
  0.0033691164782556943,
  0.8044904110141088,
];

// Cephes coefficients are in descending degree with a halved constant term.
// The Clenshaw recurrence evaluates their Chebyshev series at x/2.
function chebyshev(x: number, coefficients: readonly number[]): number {
  let current = coefficients[0], previous = 0, prior = 0;
  for (let i = 1; i < coefficients.length; i++) {
    prior = previous;
    previous = current;
    current = x * previous - prior + coefficients[i];
  }
  return 0.5 * (current - prior);
}

// Evaluate log I0 through its scaled expansion, without exp(x) overflow.
function logBesselI0(x: number): number {
  const ax = Math.abs(x);
  if (ax <= 8) return ax + Math.log(chebyshev(ax / 2 - 2, SMALL));
  return ax - 0.5 * Math.log(ax) + Math.log(chebyshev(32 / ax - 2, LARGE));
}

module.exports = { logBesselI0 };
