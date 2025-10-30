"use client"

import { useEffect, useRef } from 'react'
import './galaxy-background.css'

// 🌟 완전히 새로운 WebGL Galaxy 배경 (화려한 파티클, 강렬한 색상)
export default function GalaxyBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
    if (!gl) {
      // WebGL을 사용할 수 없으면 Canvas 2D로 폴백
      fallbackTo2D(canvas)
      return
    }

    let animationFrameId: number
    let time = 0

    const resize = () => {
      const scale = window.devicePixelRatio || 1
      canvas.width = window.innerWidth * scale
      canvas.height = window.innerHeight * scale
      canvas.style.width = window.innerWidth + 'px'
      canvas.style.height = window.innerHeight + 'px'
      gl.viewport(0, 0, canvas.width, canvas.height)
    }
    resize()
    window.addEventListener('resize', resize)

    // 버텍스 셰이더
    const vertexShaderSource = `
      attribute vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `

    // 프래그먼트 셰이더 (화려한 은하 효과)
    const fragmentShaderSource = `
      precision highp float;
      uniform float time;
      uniform vec2 resolution;

      // 별 파티클 생성
      float star(vec2 uv, float flicker) {
        float d = length(uv);
        float brightness = 0.05 / d;
        brightness *= flicker;
        brightness = pow(brightness, 1.5);
        return brightness;
      }

      // 해시 함수
      float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
      }

      // 노이즈 함수
      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float a = hash(i);
        float b = hash(i + vec2(1.0, 0.0));
        float c = hash(i + vec2(0.0, 1.0));
        float d = hash(i + vec2(1.0, 1.0));
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
      }

      void main() {
        vec2 uv = (gl_FragCoord.xy - resolution * 0.5) / resolution.y;
        
        // 밝은 흰색 배경 그라데이션
        float bgGradient = length(uv) * 0.3;
        vec3 bgColor = vec3(1.0 - bgGradient * 0.1);
        
        vec3 color = bgColor;
        
        // 여러 레이어의 별 생성 (더 화려하게)
        for(float layer = 0.0; layer < 4.0; layer++) {
          float layerDepth = layer / 4.0;
          float scale = mix(3.0, 15.0, layerDepth);
          vec2 layerUV = uv * scale + time * (0.02 + layerDepth * 0.03);
          
          vec2 gv = fract(layerUV) - 0.5;
          vec2 id = floor(layerUV);
          
          float n = hash(id + layer * 10.0);
          float size = fract(n * 345.32);
          
          // 반짝임 효과
          float flicker = 0.5 + 0.5 * sin(time * (2.0 + n * 3.0) + n * 6.28);
          
          // 별 위치 랜덤화
          vec2 offset = vec2(
            fract(n * 123.45) - 0.5,
            fract(n * 678.90) - 0.5
          ) * 0.5;
          
          float starBrightness = star(gv - offset, flicker) * size;
          
          // 화려한 색상 (파랑, 보라, 핑크, 하늘색)
          vec3 starColor;
          if (n < 0.25) {
            starColor = vec3(0.4, 0.7, 1.0); // 밝은 파랑
          } else if (n < 0.5) {
            starColor = vec3(0.8, 0.4, 1.0); // 밝은 보라
          } else if (n < 0.75) {
            starColor = vec3(1.0, 0.5, 0.8); // 밝은 핑크
          } else {
            starColor = vec3(0.5, 0.9, 1.0); // 밝은 하늘색
          }
          
          color += starColor * starBrightness * (1.0 - layerDepth * 0.3);
        }
        
        // 성운 효과 (부드러운 구름)
        float nebula1 = noise(uv * 2.0 + time * 0.05);
        float nebula2 = noise(uv * 3.0 - time * 0.03);
        vec3 nebulaColor1 = vec3(0.6, 0.8, 1.0) * nebula1 * 0.15; // 연한 파랑
        vec3 nebulaColor2 = vec3(0.9, 0.6, 1.0) * nebula2 * 0.12; // 연한 보라
        color += nebulaColor1 + nebulaColor2;
        
        // 빛나는 중심 (은하 중심부)
        float centerGlow = 1.0 - length(uv) * 0.8;
        centerGlow = pow(max(centerGlow, 0.0), 3.0);
        vec3 centerColor = vec3(0.7, 0.85, 1.0) * centerGlow * 0.2;
        color += centerColor;
        
        gl_FragColor = vec4(color, 1.0);
      }
    `

    // 셰이더 컴파일
    const createShader = (type: number, source: string) => {
      const shader = gl.createShader(type)
      if (!shader) return null
      gl.shaderSource(shader, source)
      gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader))
        gl.deleteShader(shader)
        return null
      }
      return shader
    }

    const vertexShader = createShader(gl.VERTEX_SHADER, vertexShaderSource)
    const fragmentShader = createShader(gl.FRAGMENT_SHADER, fragmentShaderSource)

    if (!vertexShader || !fragmentShader) {
      fallbackTo2D(canvas)
      return
    }

    // 프로그램 생성
    const program = gl.createProgram()
    if (!program) {
      fallbackTo2D(canvas)
      return
    }
    
    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program))
      fallbackTo2D(canvas)
      return
    }

    gl.useProgram(program)

    // 전체 화면 사각형
    const vertices = new Float32Array([
      -1, -1,
      1, -1,
      -1, 1,
      1, 1,
    ])

    const buffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    const positionLocation = gl.getAttribLocation(program, 'position')
    gl.enableVertexAttribArray(positionLocation)
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0)

    const timeLocation = gl.getUniformLocation(program, 'time')
    const resolutionLocation = gl.getUniformLocation(program, 'resolution')

    const animate = () => {
      time += 0.016 // ~60fps
      
      gl.clearColor(1.0, 1.0, 1.0, 1.0)
      gl.clear(gl.COLOR_BUFFER_BIT)

      gl.uniform1f(timeLocation, time)
      gl.uniform2f(resolutionLocation, canvas.width, canvas.height)

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

      animationFrameId = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(animationFrameId)
      gl.getExtension('WEBGL_lose_context')?.loseContext()
    }
  }, [])

  // Canvas 2D 폴백
  function fallbackTo2D(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationFrameId: number
    let time = 0

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    // 화려한 파티클 생성 (더 많이, 더 다채롭게)
    const particles: Array<{
      x: number
      y: number
      size: number
      color: string
      speed: number
      twinkleOffset: number
    }> = []

    for (let i = 0; i < 300; i++) {
      const colors = [
        'rgba(100, 180, 255, ', // 밝은 파랑
        'rgba(200, 100, 255, ', // 밝은 보라
        'rgba(255, 130, 200, ', // 밝은 핑크
        'rgba(130, 230, 255, ', // 밝은 하늘색
        'rgba(150, 200, 255, ', // 연한 파랑
      ]
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 3 + 1,
        color: colors[Math.floor(Math.random() * colors.length)],
        speed: Math.random() * 0.5 + 0.2,
        twinkleOffset: Math.random() * Math.PI * 2,
      })
    }

    const animate = () => {
      time += 0.016

      // 밝은 배경 그라데이션
      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
      gradient.addColorStop(0, '#ffffff')
      gradient.addColorStop(0.5, '#f5f8ff')
      gradient.addColorStop(1, '#e8f0ff')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // 중심 빛 효과
      const centerGradient = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 0,
        canvas.width / 2, canvas.height / 2, canvas.width * 0.6
      )
      centerGradient.addColorStop(0, 'rgba(200, 220, 255, 0.3)')
      centerGradient.addColorStop(0.5, 'rgba(220, 200, 255, 0.15)')
      centerGradient.addColorStop(1, 'rgba(255, 255, 255, 0)')
      ctx.fillStyle = centerGradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // 화려한 파티클 그리기
      particles.forEach((particle) => {
        const twinkle = Math.sin(time * 3 + particle.twinkleOffset) * 0.4 + 0.6
        const opacity = twinkle * 0.9

        // 메인 파티클
        ctx.fillStyle = particle.color + opacity + ')'
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
        ctx.fill()

        // 강렬한 발광 효과
        const glowGradient = ctx.createRadialGradient(
          particle.x, particle.y, 0,
          particle.x, particle.y, particle.size * 5
        )
        glowGradient.addColorStop(0, particle.color + (opacity * 0.6) + ')')
        glowGradient.addColorStop(1, particle.color + '0)')
        ctx.fillStyle = glowGradient
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.size * 5, 0, Math.PI * 2)
        ctx.fill()

        // 파티클 이동
        particle.y += particle.speed
        if (particle.y > canvas.height) {
          particle.y = 0
          particle.x = Math.random() * canvas.width
        }
      })

      animationFrameId = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(animationFrameId)
    }
  }

  return (
    <canvas
      ref={canvasRef}
      className="galaxy-background"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
      }}
    />
  )
}
