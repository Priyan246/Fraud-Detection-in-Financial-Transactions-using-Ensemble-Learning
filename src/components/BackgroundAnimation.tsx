import { motion } from 'framer-motion';

export default function BackgroundAnimation() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10">
      {/* Large floating circles - very subtle */}
      <motion.div
        className="absolute -top-20 -left-20 w-96 h-96 rounded-full"
        style={{ background: 'hsl(var(--foreground) / 0.03)' }}
        animate={{
          y: [0, 40, 0],
          x: [0, 20, 0],
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      
      <motion.div
        className="absolute top-1/3 -right-32 w-80 h-80 rounded-full"
        style={{ background: 'hsl(var(--foreground) / 0.02)' }}
        animate={{
          y: [0, -30, 0],
          x: [0, -15, 0],
          scale: [1, 1.05, 1],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 2,
        }}
      />
      
      <motion.div
        className="absolute -bottom-20 left-1/4 w-64 h-64 rounded-full"
        style={{ background: 'hsl(var(--foreground) / 0.025)' }}
        animate={{
          y: [0, -25, 0],
          x: [0, 25, 0],
          scale: [1, 1.08, 1],
        }}
        transition={{
          duration: 18,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 4,
        }}
      />

      {/* Floating squares - very subtle */}
      <motion.div
        className="absolute top-20 right-1/4 w-16 h-16 border rotate-12"
        style={{ borderColor: 'hsl(var(--foreground) / 0.08)' }}
        animate={{
          y: [0, -20, 0],
          rotate: [12, 20, 12],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      
      <motion.div
        className="absolute bottom-1/3 left-20 w-12 h-12 border -rotate-12"
        style={{ borderColor: 'hsl(var(--foreground) / 0.06)' }}
        animate={{
          y: [0, 15, 0],
          rotate: [-12, -5, -12],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 1,
        }}
      />
      
      <motion.div
        className="absolute top-1/2 right-10 w-8 h-8 rotate-45"
        style={{ background: 'hsl(var(--foreground) / 0.05)' }}
        animate={{
          y: [0, -10, 0],
          rotate: [45, 55, 45],
        }}
        transition={{
          duration: 6,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 3,
        }}
      />

      {/* Grid dots - very subtle */}
      <div 
        className="absolute inset-0"
        style={{
          opacity: 0.04,
          backgroundImage: `radial-gradient(circle, hsl(var(--foreground)) 1px, transparent 1px)`,
          backgroundSize: '40px 40px',
        }}
      />
    </div>
  );
}
